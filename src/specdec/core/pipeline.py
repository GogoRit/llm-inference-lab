"""
Speculative Decoding Pipeline

Orchestrates the speculative decoding loop:
1. Use draft model to propose up to K tokens
2. Verify with base model; accept longest prefix
3. If accepted_len == 0, advance by 1 token using base model
4. Repeat until max_tokens or stop condition
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import yaml

from ..cache.kv_cache_manager import SafeKVCacheManager
from ..models.fake_lm import create_fake_lm
from ..models.hf_wrappers import create_tiny_hf_wrapper
from ..policies.controllers import KController, create_controller
from ..policies.policies import AcceptancePolicy, create_policy
from ..utils.deterministic import ensure_deterministic
from ..utils.interfaces import LanguageModel, SpeculativeDecoder
from ..utils.token_validation import get_vocab_size, validate_and_clamp_tokens
from .kv_cache_verification import (
    compute_kv_checksum,
    debug_verify_kv_cache_step,
    verify_kv_cache_alignment,
)
from .sequence_pool import SequencePool
from .sequence_utils import (
    create_position_ids,
    pad_sequences,
    unpad_append_repad,
    unpad_sequences,
)

logger = logging.getLogger(__name__)


# filter_kv_cache_safe removed - use SafeKVCacheManager instead


def sample_bonus_token_from_logits(
    logits: torch.Tensor,
    temperature: float,
    do_sample: bool,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a bonus token from target model logits with proper sampling settings.

    This implements EQSPEC bonus token sampling: after accepting draft tokens,
    we sample exactly one token from the target model distribution at the
    mismatch position.

    Args:
        logits: [vocab_size] or [1, vocab_size] logits from target model
        temperature: Sampling temperature
        do_sample: Whether to use sampling (True) or greedy (False)
        top_p: Top-p (nucleus) sampling parameter (optional)
        top_k: Top-k sampling parameter (optional)
        vocab_size: Vocabulary size for validation (optional)

    Returns:
        bonus_token: [1] tensor with sampled token ID
    """
    # Ensure logits are 2D [1, vocab_size]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # [1, vocab_size]

    # Validate logits shape
    if logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, vocab_size], got {logits.shape}")

    # Validate vocab size if provided
    if vocab_size is not None:
        if logits.shape[1] != vocab_size:
            logger.warning(
                f"Logits vocab size {logits.shape[1]} != expected {vocab_size}, "
                "clamping tokens to valid range"
            )

    # Apply temperature
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering if specified
    if top_k is not None and top_k > 0:
        # Get top-k logits, set others to -inf
        top_k_logits, top_k_indices = torch.topk(
            logits, min(top_k, logits.shape[1]), dim=-1
        )
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
        logits = filtered_logits

    # Apply top-p (nucleus) filtering if specified
    if top_p is not None and top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # Compute cumulative probabilities
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Find cutoff point
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Create mask for tokens to keep
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample or use argmax
    if do_sample:
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)

        # Safety check for invalid probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            logger.warning("Invalid probabilities detected, falling back to argmax")
            bonus_token = logits.argmax(dim=-1, keepdim=True)
        else:
            # Sample from distribution
            bonus_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
    else:
        # Greedy: use argmax
        bonus_token = logits.argmax(dim=-1, keepdim=True)  # [1, 1]

    # Ensure dtype is long
    bonus_token = bonus_token.long()

    # Validate token is in valid range
    if vocab_size is not None:
        bonus_token = bonus_token.clamp(min=0, max=vocab_size - 1)

    # Return as [1] tensor (squeeze batch dimension)
    return bonus_token.squeeze(0)  # [1]


# Import optimization modules (conditional to avoid import errors during development)
try:
    from src.benchmarks.profiler import create_profiler
    from src.metrics.structured_profiler import create_structured_profiler
    from src.optimization import (
        amp_context,
        create_optimization_manager,
        create_optimized_tokenizer,
        select_device_dtype,
    )
    from src.scheduler import create_speculative_scheduler
except ImportError:
    # Fallback for development - create dummy functions
    def create_optimization_manager(*args, **kwargs):  # type: ignore[misc]
        return None

    def create_optimized_tokenizer(*args, **kwargs):  # type: ignore[misc]
        return None

    def create_profiler(*args, **kwargs):  # type: ignore[misc]
        return None

    def create_structured_profiler(*args, **kwargs):  # type: ignore[misc]
        # Return dummy structured profiler for development
        class DummyProfiler:
            enable_profiling = False

            def record_step(self, *args, **kwargs):
                pass

            def record_kv_append_time(self, *args, **kwargs):
                pass

        return DummyProfiler()

    def create_speculative_scheduler(*args, **kwargs):  # type: ignore[misc]
        return None

    def select_device_dtype(device="auto"):  # type: ignore[misc]
        return device, torch.float32, False

    def amp_context(device, dtype):  # type: ignore[misc]
        return torch.no_grad()


def _get_max_position_embeddings(model: Any) -> Optional[int]:
    """
    Extract max_position_embeddings from a language model.

    Args:
        model: Language model instance (HFWrapper, FakeLM, etc.)

    Returns:
        max_position_embeddings if available, None otherwise
    """
    # Try to get from HF model config
    if hasattr(model, "_model") and hasattr(model._model, "config"):
        config = model._model.config
        if hasattr(config, "max_position_embeddings"):
            return getattr(config, "max_position_embeddings")
        # Some models use n_positions instead
        if hasattr(config, "n_positions"):
            return getattr(config, "n_positions")

    # Try direct access to model.config
    if hasattr(model, "model") and hasattr(model.model, "config"):
        config = model.model.config
        if hasattr(config, "max_position_embeddings"):
            return getattr(config, "max_position_embeddings")
        if hasattr(config, "n_positions"):
            return getattr(config, "n_positions")

    return None


class SpeculativePipeline(SpeculativeDecoder):
    """Main pipeline for speculative decoding with dependency injection."""

    def __init__(
        self,
        base_lm: Optional[LanguageModel] = None,
        draft_lm: Optional[LanguageModel] = None,
        config_path: Optional[str] = None,
        base_model: Optional[str] = None,
        draft_model: Optional[str] = None,
        max_draft: int = 4,
        device: str = "auto",
        seed: Optional[int] = None,
        implementation: Optional[str] = None,
        force_device: Optional[str] = None,
        policy: str = "longest_prefix",
        policy_params: Optional[Dict[str, Any]] = None,
        controller: str = "fixed",
        controller_params: Optional[Dict[str, Any]] = None,
        draft_mode: str = "vanilla",
        enable_optimization: bool = True,
        enable_profiling: bool = False,
        profile_dir: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        max_seq_len_cap: int = 4096,
    ):
        """
        Initialize the speculative decoding pipeline.

        Args:
            base_lm: Base language model (if None, will be created)
            draft_lm: Draft language model (if None, will be created)
            config_path: Path to YAML configuration file
            base_model: Base model name (overrides config)
            draft_model: Draft model name (overrides config)
            max_draft: Maximum draft tokens per step (overrides config)
            device: Device to run on (overrides config)
            seed: Random seed for reproducibility
            implementation: Implementation type ("fake" or "hf")
            force_device: Force both models to same device ("cpu", "mps")
            policy: Acceptance policy name ("longest_prefix",
                   "conf_threshold", "topk_agree", "typical")
            policy_params: Parameters for the acceptance policy
            controller: K controller type ("fixed", "adaptive")
            controller_params: Parameters for the K controller
            draft_mode: Draft mode ("vanilla", "medusa", "eagle")
            enable_optimization: Whether to enable performance optimizations
            enable_profiling: Whether to enable profiling
            profile_dir: Directory to save profiling traces
            max_seq_len: Explicit max_seq_len for KV cache (overrides calculated value)
            max_seq_len_cap: Hard cap on calculated max_seq_len to prevent OOM (default: 4096)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)

        # Override config with provided parameters
        if base_model:
            self.config["base_model"] = base_model
        if draft_model:
            self.config["draft_model"] = draft_model
        if max_draft:
            self.config["max_draft"] = max_draft
        if device is not None and device != "auto":
            self.config["device"] = device
        if implementation:
            self.config["implementation"] = implementation
        if force_device:
            self.config["force_device"] = force_device
        if draft_mode:
            self.config["draft_mode"] = draft_mode

        # Set random seed if provided
        if seed is not None:
            self._set_seed(seed)
            self.config["seed"] = seed

        self.max_draft = self.config["max_draft"]

        # Use centralized device/dtype selection
        device_pref = self.config["device"]
        if force_device:
            device_pref = force_device
        self.device, self.dtype, self.amp_enabled = select_device_dtype(device_pref)

        self.implementation = self.config.get("implementation", "fake")
        self.force_device = self.config.get("force_device")

        # Phase 3D: ensure deterministic mode if requested via env
        ensure_deterministic()

        # Phase 3D: structured profiler (off unless SPECDEC_PROFILE=1)
        self.structured_profiler = create_structured_profiler(
            device=self.device if isinstance(self.device, str) else str(self.device),
            enable_profiling=False,
        )

        # CUDA graph capture removed - incompatible with dynamic speculative decoding
        # Set attributes to False to prevent AttributeError in generate()
        self.enable_cuda_graph = False
        self.cuda_graph_warmup_steps = 0
        self.cuda_graph_warmup_done = False
        self.cuda_graph_captured = False
        self.graph_input_tensor = None
        # Use CUDA streams for parallelization instead

        # Set deterministic flags
        self.deterministic = self.config.get("deterministic", False)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Log startup configuration
        self._log_startup_config()

        # Initialize models with dependency injection
        self.base_lm = base_lm or self._create_base_model()
        self.draft_lm = draft_lm or self._create_draft_model()

        # CRITICAL: Initialize KV cache manager AFTER models are created
        # to get max_position_embeddings from model configs
        # Safety: Use explicit max_seq_len if provided, otherwise calculate with cap
        if max_seq_len is not None:
            # User explicitly provided max_seq_len - respect it
            self.logger.info(f"Using explicit max_seq_len={max_seq_len} from parameter")
            final_max_seq_len = max_seq_len
        else:
            # Calculate from model configs with VRAM safety clamp
            final_max_seq_len = self._calculate_max_seq_len(max_allowed=max_seq_len_cap)
        self.kv_cache_manager = SafeKVCacheManager(
            device=str(self.device),
            max_seq_len=final_max_seq_len,
        )
        self.logger.info(
            f"Initialized KV cache manager with max_seq_len={final_max_seq_len}"
        )

        # Check if baseline mode (no speculative decoding)
        self.speculative_enabled = self.draft_lm is not None
        if not self.speculative_enabled:
            self.logger.info(
                "Non-speculative baseline mode enabled: using only base model "
                "for standard autoregressive decoding"
            )

        # Initialize policy and controller (only needed for speculative mode)
        if self.speculative_enabled:
            self.policy = self._create_policy(policy, policy_params)
            self.controller = self._create_controller(controller, controller_params)
        else:
            # Create dummy policy/controller to avoid AttributeError
            # (they won't be used in baseline mode)
            self.policy = self._create_policy(policy, policy_params)
            self.controller = self._create_controller(controller, controller_params)

        # Initialize speculative scheduler
        self.scheduler = self._create_scheduler()

        # Check tokenizer compatibility
        self._check_compatibility()

        # Initialize optimization and profiling
        self.enable_optimization = enable_optimization
        self.enable_profiling = enable_profiling

        if self.enable_optimization:
            self.optimization_manager = create_optimization_manager(
                device=self.device,
                mixed_precision=True,
                gradient_checkpointing=False,  # Disabled for inference
                memory_efficient=True,
                dtype=self.dtype,
            )
        else:
            self.optimization_manager = None

        if self.enable_optimization and self.optimization_manager is not None:
            # Use clean API for optimization
            try:
                if hasattr(self.base_lm, "optimize"):
                    logger.info("Optimizing base model...")
                    self.base_lm.optimize(self.optimization_manager)
                    logger.info("Base model optimization completed")
                # Note: Base model optimization handled via optimize() method above

                if (
                    self.speculative_enabled
                    and self.draft_lm is not None
                    and hasattr(self.draft_lm, "optimize")
                ):
                    logger.info("Optimizing draft model...")
                    self.draft_lm.optimize(self.optimization_manager)
                    logger.info("Draft model optimization completed")
                # Note: Draft model optimization handled via optimize() method above
            except Exception as e:
                logger.error(f"Model optimization failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
                # Continue without optimization rather than failing
                self.optimization_manager = None

        if self.enable_profiling:
            self.profiler = create_profiler(
                enable_profiling=True,
                profile_dir=profile_dir,
                memory_tracking=True,
                device=self.force_device or self.device,
            )
        else:
            self.profiler = None

        # Metrics tracking
        self.metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_steps": 0,
            "total_verification_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
            "kv_appended_tokens_total": 0,
            "kv_append_time_ms": 0.0,
        }

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "base_model": "facebook/opt-125m",
            "draft_model": "distilgpt2",
            "max_draft": 4,
            "temperature": 0.7,
            "do_sample": True,
            "device": "auto",
            "seed": 1234,
            "max_new_tokens": 64,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "implementation": "fake",
            "draft_mode": "vanilla",
            "medusa": {
                "enabled": False,
                "num_heads": 2,
                "head_init": "tie",
                "temperature": 0.7,
                "top_p": 1.0,
            },
            "eagle": {
                "enabled": False,
                "alpha": 0.7,
                "max_draft": 2,
            },
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                default_config.update(config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def _calculate_max_seq_len(self, max_allowed: int = 4096) -> int:
        """
        Calculate maximum sequence length from model configs with VRAM safety clamp.

        Uses max_position_embeddings from base and draft models, taking the maximum.
        Applies a hard cap to prevent OOM on constrained hardware (e.g., T4 16GB).
        Falls back to 2048 if not available.

        Args:
            max_allowed: Hard cap on max_seq_len to prevent OOM (default: 4096)

        Returns:
            Maximum sequence length for KV cache buffers (clamped to max_allowed)
        """
        max_seq_lens = []

        # Get max_position_embeddings from base model
        if self.base_lm is not None:
            base_max_seq = _get_max_position_embeddings(self.base_lm)
            if base_max_seq is not None:
                max_seq_lens.append(base_max_seq)
                self.logger.debug(f"Base model max_position_embeddings: {base_max_seq}")

        # Get max_position_embeddings from draft model
        if self.draft_lm is not None:
            draft_max_seq = _get_max_position_embeddings(self.draft_lm)
            if draft_max_seq is not None:
                max_seq_lens.append(draft_max_seq)
                self.logger.debug(
                    f"Draft model max_position_embeddings: {draft_max_seq}"
                )

        # Use maximum of both models, or fallback to 2048
        if max_seq_lens:
            model_max_seq_len = max(max_seq_lens)
            # CRITICAL: Clamp to max_allowed to prevent OOM on constrained hardware
            if model_max_seq_len > max_allowed:
                self.logger.warning(
                    f"Clamping max_seq_len from {model_max_seq_len} to {max_allowed} "
                    f"to save VRAM (model config exceeds hardware limits)"
                )
                max_seq_len = max_allowed
            else:
                max_seq_len = model_max_seq_len
                self.logger.info(f"Using max_seq_len={max_seq_len} from model configs")
        else:
            # Fallback to safe default (already within max_allowed)
            max_seq_len = min(2048, max_allowed)
            self.logger.warning(
                f"Could not determine max_position_embeddings from models. "
                f"Using default max_seq_len={max_seq_len}"
            )

        return max_seq_len

    def _log_startup_config(self) -> None:
        """Log startup configuration summary."""
        device = self.force_device or self.device
        dtype = "float16" if device == "mps" else "float32"

        self.logger.info(
            f"Startup config: impl={self.implementation}, device={device}, "
            f"dtype={dtype}, base_model={self.config['base_model']}, "
            f"draft_model={self.config['draft_model']}, max_draft={self.max_draft}, "
            f"max_tokens={self.config.get('max_new_tokens', 64)}"
        )

        # Log kernel information
        try:
            from kernels import get_kernel_info

            kernel_info = get_kernel_info()
            self.logger.info(
                f"Kernel backends: verify={kernel_info['verify_backend']}, "
                f"kv_append={kernel_info['kv_append_backend']}"
            )
        except ImportError:
            self.logger.info("Kernels not available, using fallback implementation")

    def _create_base_model(self) -> LanguageModel:
        """Create the base language model based on implementation."""
        if self.implementation == "fake":
            return create_fake_lm(
                model_name=f"fake-base-{self.config['base_model']}",
                vocab_size=1000,
                device="cpu",
                seed=self.config.get("seed"),
            )
        elif self.implementation == "hf":
            # Use tiny models for HF implementation
            device = self.force_device or self.device
            return create_tiny_hf_wrapper(
                model_name=self.config["base_model"],
                device=device,
                max_memory_mb=500,
            )
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _create_draft_model(self) -> Optional[LanguageModel]:
        """Create the draft language model based on implementation."""
        # Check if baseline mode is requested (no draft model)
        draft_model_name = self.config.get("draft_model", "")
        if draft_model_name in (None, "", "none", "NONE"):
            self.logger.info("Baseline mode: No draft model will be loaded")
            return None

        if self.implementation == "fake":
            return create_fake_lm(
                model_name=f"fake-draft-{self.config['draft_model']}",
                vocab_size=1000,
                device="cpu",
                seed=self.config.get("seed"),
                # Don't use acceptance rate for deterministic behavior
            )
        elif self.implementation == "hf":
            # Use tiny models for HF implementation with shared tokenizer
            device = self.force_device or self.device
            base_lm = self.base_lm
            shared_tokenizer = None
            if hasattr(base_lm, "_tokenizer"):
                shared_tokenizer = base_lm._tokenizer

            return create_tiny_hf_wrapper(
                model_name=self.config["draft_model"],
                device=device,
                max_memory_mb=500,
                tokenizer=shared_tokenizer,
            )
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        import random

        random.seed(seed)
        self.logger.info(f"Set random seed to {seed}")

    def _select_device(self, device: str) -> str:
        """Select the best available device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _create_policy(
        self, policy_name: str, policy_params: Optional[Dict[str, Any]]
    ) -> AcceptancePolicy:
        """Create acceptance policy."""
        params = policy_params or {}
        return create_policy(policy_name, **params)

    def _create_controller(
        self, controller_type: str, controller_params: Optional[Dict[str, Any]]
    ) -> KController:
        """Create K controller."""
        params = controller_params or {}
        return create_controller(controller_type, **params)

    def _create_scheduler(self):
        """Create speculative scheduler."""
        device = self.force_device or self.device
        if create_speculative_scheduler:
            return create_speculative_scheduler(
                device=device,
                enable_multi_stream=device == "cuda",
                enable_batched_verification=True,
            )
        return None

    def _check_compatibility(self) -> None:
        """Check tokenizer compatibility between draft and base models."""
        base_info = self.base_lm.get_tokenizer_info()

        # Baseline mode: no draft model, skip cross-model checks
        if self.draft_lm is None or not getattr(self, "speculative_enabled", True):
            # In baseline mode, just use base model tokenizer info
            if hasattr(self, "tokenizer_info"):
                self.tokenizer_info = base_info
            self.logger.debug("Baseline mode: skipping draft model compatibility check")
            return

        # Existing speculative path: requires draft_lm
        draft_info = self.draft_lm.get_tokenizer_info()

        # Check if tokenizers are compatible
        compatible = (
            base_info["vocab_size"] == draft_info["vocab_size"]
            and base_info["pad_token_id"] == draft_info["pad_token_id"]
            and base_info["eos_token_id"] == draft_info["eos_token_id"]
        )

        if not compatible:
            self.logger.warning(
                f"Tokenizer incompatibility detected: "
                f"base={base_info}, draft={draft_info}"
            )
            self.logger.warning(
                "Different tokenizer families detected. This may reduce "
                "acceptance rates."
            )

        # Store tokenizer info (use base as canonical)
        if hasattr(self, "tokenizer_info"):
            self.tokenizer_info = base_info

    def _generate_medusa_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Generate tokens using Medusa draftor."""
        if self.implementation == "fake":
            # For FakeLM, use vanilla generation
            self.logger.debug("Medusa mode: using vanilla generation for FakeLM")
            draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                input_ids,
                max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs,
            )
            draft_info = {"mode": "medusa_fake"}
        else:
            # For HF models, implement actual Medusa logic
            self.logger.debug("Medusa mode: implementing actual Medusa logic")
            draft_tokens, draft_logits, draft_info = self._run_medusa_hf(
                input_ids, max_new_tokens, temperature, do_sample, **kwargs
            )
        return draft_tokens, draft_logits, draft_info

    def _generate_eagle_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Generate tokens using EAGLE draftor."""
        if self.implementation == "fake":
            # For FakeLM, use vanilla generation
            self.logger.debug("EAGLE mode: using vanilla generation for FakeLM")
            draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                input_ids,
                max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs,
            )
            draft_info = {"mode": "eagle_fake"}
        else:
            # For HF models, implement actual EAGLE logic
            self.logger.debug("EAGLE mode: implementing actual EAGLE logic")
            draft_tokens, draft_logits, draft_info = self._run_eagle_hf(
                input_ids, max_new_tokens, temperature, do_sample, **kwargs
            )
        return draft_tokens, draft_logits, draft_info

    def _run_medusa_hf(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Run actual Medusa logic for HF models."""
        import time

        start_time = time.time()

        # Get base model and tokenizer
        base_model = self.base_lm.model  # type: ignore
        tokenizer = self.base_lm.tokenizer  # type: ignore

        # Get hidden states from base model
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get last hidden state
            last_hidden_state = outputs.hidden_states[
                -1
            ]  # [batch_size, seq_len, hidden_size]
            current_hidden = last_hidden_state[
                :, -1:, :
            ]  # [batch_size, 1, hidden_size]

            # Create multiple prediction heads (simplified Medusa)
            num_heads = self.config.get("medusa", {}).get("num_heads", 2)
            hidden_size = last_hidden_state.shape[-1]
            vocab_size = tokenizer.vocab_size

            # Create simple linear heads
            heads = []
            device = input_ids.device
            for i in range(num_heads):
                head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
                # Initialize with small random weights
                torch.nn.init.normal_(head.weight, 0, 0.02)
                # Move to correct device
                head = head.to(device)
                heads.append(head)

            generated_tokens = []
            all_logits = []

            for step in range(max_new_tokens):
                step_tokens = []
                step_logits = []

                # Use each head to predict one token
                for head_idx in range(min(num_heads, max_new_tokens - step)):
                    head = heads[head_idx]
                    head_logits = head(current_hidden)  # [batch_size, 1, vocab_size]
                    step_logits.append(head_logits)

                    # Sample token from head logits
                    if temperature > 0:
                        head_logits = head_logits / temperature

                    probs = torch.softmax(head_logits, dim=-1)
                    next_token = torch.multinomial(
                        probs.squeeze(1), 1
                    )  # [batch_size, 1]
                    step_tokens.append(next_token)

                if not step_tokens:
                    break

                # Use first head's prediction for this step
                next_token = step_tokens[0]
                generated_tokens.append(next_token)
                all_logits.append(step_logits[0])

                # Update hidden state for next iteration (simplified)
                current_hidden = current_hidden  # Keep same hidden state for simplicity

        generation_time_ms = (time.time() - start_time) * 1000

        # Concatenate generated tokens
        if generated_tokens:
            draft_tokens = torch.cat(
                generated_tokens, dim=1
            )  # [batch_size, num_generated]
            draft_logits = torch.cat(
                all_logits, dim=1
            )  # [batch_size, num_generated, vocab_size]
        else:
            batch_size = input_ids.shape[0]
            draft_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=input_ids.device
            )
            draft_logits = torch.empty(
                batch_size, 0, vocab_size, device=input_ids.device
            )

        draft_info = {
            "mode": "medusa_hf",
            "generation_time_ms": generation_time_ms,
            "num_heads_used": len(step_tokens) if step_tokens else 0,
        }

        return draft_tokens, draft_logits, draft_info

    def _run_eagle_hf(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Run actual EAGLE logic for HF models."""
        import time

        start_time = time.time()

        # Get base model and tokenizer
        base_model = self.base_lm.model  # type: ignore
        tokenizer = self.base_lm.tokenizer  # type: ignore

        # Get EAGLE parameters
        alpha = self.config.get("eagle", {}).get("alpha", 0.7)
        max_draft = self.config.get("eagle", {}).get("max_draft", 2)

        # Get hidden states from base model
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get last hidden state
            last_hidden_state = outputs.hidden_states[
                -1
            ]  # [batch_size, seq_len, hidden_size]
            current_hidden = last_hidden_state[
                :, -1:, :
            ]  # [batch_size, 1, hidden_size]

            # Initialize state tracking
            if not hasattr(self, "_eagle_last_hidden_states"):
                self._eagle_last_hidden_states = (
                    current_hidden  # [batch_size, 1, hidden_size]
                )
            else:
                # Update with new hidden state
                self._eagle_last_hidden_states = torch.cat(
                    [self._eagle_last_hidden_states[:, -1:, :], current_hidden], dim=1
                )  # [batch_size, 2, hidden_size]

            generated_tokens = []
            all_logits = []

            # Generate up to max_draft tokens
            num_tokens_to_generate = min(max_new_tokens, max_draft)

            for step in range(num_tokens_to_generate):
                # Extrapolate next hidden state
                if self._eagle_last_hidden_states.shape[1] >= 2:
                    # We have at least 2 hidden states for extrapolation
                    h_t_minus_1 = self._eagle_last_hidden_states[
                        :, -2, :
                    ]  # [batch_size, hidden_size]
                    h_t = self._eagle_last_hidden_states[
                        :, -1, :
                    ]  # [batch_size, hidden_size]

                    # EAGLE extrapolation: h_next = h_t + alpha * (h_t - h_t_minus_1)
                    h_next = h_t + alpha * (h_t - h_t_minus_1)
                else:
                    # Fallback: use current hidden state
                    h_next = current_hidden.squeeze(1)  # [batch_size, hidden_size]

                # Get logits from language modeling head
                lm_head = base_model.lm_head
                next_logits = lm_head(h_next)  # [batch_size, vocab_size]

                # Sample next token
                next_token = torch.argmax(
                    next_logits, dim=-1, keepdim=True
                )  # [batch_size, 1]

                generated_tokens.append(next_token)
                all_logits.append(
                    next_logits.unsqueeze(1)
                )  # [batch_size, 1, vocab_size]

                # Update state for next iteration
                h_next_expanded = h_next.unsqueeze(1)  # [batch_size, 1, hidden_size]
                self._eagle_last_hidden_states = torch.cat(
                    [self._eagle_last_hidden_states, h_next_expanded], dim=1
                )  # [batch_size, 3, hidden_size]

                # Keep only last 2 states for next extrapolation
                if self._eagle_last_hidden_states.shape[1] > 2:
                    self._eagle_last_hidden_states = self._eagle_last_hidden_states[
                        :, -2:, :
                    ]

        generation_time_ms = (time.time() - start_time) * 1000

        # Concatenate generated tokens
        if generated_tokens:
            draft_tokens = torch.cat(
                generated_tokens, dim=1
            )  # [batch_size, num_generated]
            draft_logits = torch.cat(
                all_logits, dim=1
            )  # [batch_size, num_generated, vocab_size]
        else:
            batch_size = input_ids.shape[0]
            vocab_size = tokenizer.vocab_size
            draft_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=input_ids.device
            )
            draft_logits = torch.empty(
                batch_size, 0, vocab_size, device=input_ids.device
            )

        draft_info = {
            "mode": "eagle_hf",
            "generation_time_ms": generation_time_ms,
            "alpha": alpha,
            "extrapolation_steps": len(generated_tokens),
        }

        return draft_tokens, draft_logits, draft_info

    # CUDA graph capture methods removed - incompatible with dynamic speculative decoding

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            do_sample: Whether to use sampling (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated text and metrics
        """
        start_time = time.time()

        # Start profiling if enabled
        if self.profiler:
            self.profiler.start_memory_tracking()

        # Use provided parameters or fall back to config
        max_tokens = max_tokens or self.config["max_new_tokens"]
        temperature = temperature or self.config["temperature"]
        do_sample = do_sample if do_sample is not None else self.config["do_sample"]

        self.logger.debug(f"max_tokens: {max_tokens}, type: {type(max_tokens)}")
        self.logger.debug(f"temperature: {temperature}, type: {type(temperature)}")
        self.logger.debug(f"do_sample: {do_sample}, type: {type(do_sample)}")

        # Reset metrics
        self.metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_steps": 0,
            "total_verification_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
            "kv_appended_tokens_total": 0,
            "kv_append_time_ms": 0.0,
        }

        # Reset KV cache manager
        self.kv_cache_manager.reset()
        self.kv_cache_manager.set_batch_size(1)  # Single prompt

        # Clear KV cache at start of new generation
        if hasattr(self.base_lm, "clear_kv_cache"):
            self.base_lm.clear_kv_cache()
        if hasattr(self.draft_lm, "clear_kv_cache"):
            self.draft_lm.clear_kv_cache()

        try:
            # Use optimization context if available, otherwise use AMP context
            if self.optimization_manager:
                optimization_context = (
                    self.optimization_manager.get_optimization_context()
                )
            else:
                optimization_context = (
                    amp_context(self.device, self.dtype)
                    if self.amp_enabled
                    else torch.no_grad()
                )

            with optimization_context:
                # Tokenize input prompt
                input_ids = self.base_lm.encode(prompt)
                # Move to device if needed (but only if device is not "auto")
                if (
                    self.device != "auto"
                    and self.device != "cpu"
                    and input_ids.device == torch.device("cpu")
                ):
                    input_ids = input_ids.to(self.device)

            # Initialize generation state
            generated_tokens: List[int] = []
            current_input = input_ids.clone()
            step = 0

            self.logger.info(
                f'Starting speculative decoding: prompt="{prompt[:50]}...", '
                f"max_tokens={max_tokens}"
            )

            while (
                len(generated_tokens) < max_tokens and step < max_tokens * 2
            ):  # Safety limit
                self.logger.debug(
                    f"Loop condition: len(generated_tokens)={len(generated_tokens)} "
                    f"< max_tokens={max_tokens} and step={step} "
                    f"< max_tokens*2={max_tokens * 2}"
                )
                step += 1
                step_start = time.time()

                # Get K from controller
                context = {
                    "step": step,
                    "generated_tokens": len(generated_tokens),
                    "acceptance_rate": (
                        self.metrics["total_accepted"]
                        / max(self.metrics["total_proposed"], 1)
                    ),
                }
                self.logger.debug(f"Context: {context}")
                k = self.controller.get_k(step, context)
                self.logger.debug(f"K from controller: {k}, type: {type(k)}")

                # Step 1: Generate draft tokens based on draft mode
                draft_start = time.time()
                self.logger.debug(
                    f"Generating draft tokens with k={k}, "
                    f"temperature={temperature}, do_sample={do_sample}, "
                    f"draft_mode={self.config.get('draft_mode', 'vanilla')}"
                )

                draft_mode = self.config.get("draft_mode", "vanilla")
                if draft_mode == "vanilla":
                    # Use existing draft model
                    draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                        current_input,
                        max_new_tokens=k,
                        temperature=temperature,
                        do_sample=do_sample,
                        **kwargs,
                    )
                elif draft_mode == "medusa":
                    # Use Medusa draftor
                    draft_tokens, draft_logits, draft_info = (
                        self._generate_medusa_tokens(
                            current_input, k, temperature, do_sample, **kwargs
                        )
                    )
                elif draft_mode == "eagle":
                    # Use EAGLE draftor
                    draft_tokens, draft_logits, draft_info = (
                        self._generate_eagle_tokens(
                            current_input, k, temperature, do_sample, **kwargs
                        )
                    )
                else:
                    raise ValueError(f"Unknown draft mode: {draft_mode}")

                draft_time_ms = (time.time() - draft_start) * 1000
                self.logger.debug(
                    f"Draft tokens shape: {draft_tokens.shape}, "
                    f"logits shape: {draft_logits.shape}"
                )

                # Step 2: Verify with base model using scheduler
                # CUDA graph capture removed - incompatible with dynamic speculative decoding
                # All paths use standard verification now

                # Single validation point before model call
                base_vocab_size = get_vocab_size(self.base_lm)
                if base_vocab_size is not None:
                    current_input = validate_and_clamp_tokens(
                        current_input, base_vocab_size, "base_input"
                    )

                # Removed unnecessary sync - CUDA events handle synchronization

                # Use scheduler or direct call
                if self.scheduler:
                    base_tokens, base_logits, verify_info = (
                        self.scheduler.schedule_verification(
                            self.base_lm,
                            draft_tokens,
                            current_input,
                            temperature=temperature,
                            do_sample=do_sample,
                            **kwargs,
                        )
                    )
                    verify_time_ms = verify_info.get("verification_time_ms", 0.0)
                else:
                    verify_start = time.time()
                    base_tokens, base_logits = self.base_lm.generate_tokens(
                        current_input,
                        max_new_tokens=k,
                        temperature=temperature,
                        do_sample=do_sample,
                        **kwargs,
                    )
                    verify_time_ms = (time.time() - verify_start) * 1000
                    verify_info = {
                        "verification_time_ms": verify_time_ms,
                        "method": "eager",
                    }

                # Step 3: Apply acceptance policy
                if self.scheduler:
                    accepted_len, policy_info = self.scheduler.apply_acceptance_policy(
                        self.policy,
                        draft_tokens,
                        base_tokens,
                        draft_logits,
                        base_logits,
                    )
                else:
                    accepted_len, policy_info = self.policy.accept_tokens(
                        draft_tokens, base_tokens, draft_logits, base_logits
                    )
                accepted_tokens = (
                    draft_tokens[:, :accepted_len]
                    if accepted_len > 0
                    else torch.empty(draft_tokens.shape[0], 0, dtype=draft_tokens.dtype)
                )

                # Log acceptance details (only if debug flag enabled)
                if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                    proposed_count = draft_tokens.shape[1]
                    rejected_count = proposed_count - accepted_len
                    self.logger.debug(
                        f"[SCHED] Step {step} | "
                        f"K={proposed_count} | "
                        f"accepted={accepted_len} | "
                        f"rejected={rejected_count} | "
                        f"draft={draft_time_ms:.1f}ms | "
                        f"verify={verify_time_ms:.1f}ms"
                    )

                # Update metrics
                self.metrics["total_proposed"] += draft_tokens.shape[1]
                self.metrics["total_accepted"] += accepted_len
                self.metrics["total_verification_time_ms"] += verify_time_ms
                self.metrics["total_generation_time_ms"] += draft_time_ms

                # Log step details
                self.logger.info(
                    f"Step {step}: K={k}, proposed={draft_tokens.shape[1]}, "
                    f"accepted={accepted_len}, t_draft={draft_time_ms:.1f}ms, "
                    f"t_verify={verify_time_ms:.1f}ms, "
                    f"total={len(generated_tokens)}/{max_tokens}"
                )

                # Step 3: Handle results
                if accepted_len > 0:
                    # KV cache integration: append accepted tokens' KV
                    # to base model cache (supports partial acceptance)
                    if (
                        hasattr(self.base_lm, "supports_kv_append")
                        and self.base_lm.supports_kv_append()
                    ):
                        kv_append_start = time.time()
                        try:
                            # Get the base model's last generated KV cache
                            if hasattr(self.base_lm, "get_last_generated_kv"):
                                base_kv = self.base_lm.get_last_generated_kv()
                                if (
                                    base_kv is not None
                                    and accepted_len <= base_kv.seq_len
                                ):
                                    # Slice to accepted length (partial KV cache reuse)
                                    # This enables reuse even when only some tokens are accepted
                                    accepted_kv = base_kv.slice_prefix(accepted_len)
                                    # Append to base model's cache
                                    self.base_lm.append_kv_cache(accepted_kv)
                                    # Update metrics
                                    self.metrics[
                                        "kv_appended_tokens_total"
                                    ] += accepted_len
                                    kv_append_time_ms = (
                                        time.time() - kv_append_start
                                    ) * 1000
                                    self.metrics[
                                        "kv_append_time_ms"
                                    ] += kv_append_time_ms
                                    # Phase 3D: structured profiling hook for KV append
                                    try:
                                        if getattr(
                                            self.structured_profiler,
                                            "enable_profiling",
                                            False,
                                        ):
                                            self.structured_profiler.record_kv_append_time(
                                                kv_append_time_ms
                                            )
                                    except Exception:
                                        pass
                                    self.logger.debug(
                                        f"Appended {accepted_len} tokens to KV cache "
                                        f"in {kv_append_time_ms:.2f}ms"
                                    )
                        except Exception as e:
                            self.logger.warning(
                                f"KV cache append failed, continuing without cache: {e}"
                            )

                    # Accept the proposed tokens (but respect max_tokens limit)
                    if accepted_tokens.numel() > 0:
                        # Calculate how many tokens we can still accept
                        remaining_tokens = max_tokens - len(generated_tokens)
                        if remaining_tokens > 0:
                            # Only accept up to the remaining limit
                            tokens_to_accept = min(
                                accepted_tokens.shape[1], remaining_tokens
                            )
                            accepted_tokens_limited = accepted_tokens[
                                :, :tokens_to_accept
                            ]
                            generated_tokens.extend(
                                accepted_tokens_limited[0].cpu().tolist()
                            )
                            current_input = torch.cat(
                                [current_input, accepted_tokens_limited], dim=1
                            )

                    self.logger.info(
                        f"Step {step}: accepted {accepted_len}/"
                        f"{draft_tokens.shape[1]} tokens, "
                        f"total: {len(generated_tokens)}/{max_tokens}"
                    )
                else:
                    # No tokens accepted - fallback: generate one token with base model
                    # (if we haven't reached max_tokens)
                    remaining_tokens = max_tokens - len(generated_tokens)
                    if remaining_tokens > 0:
                        fallback_tokens, _ = self.base_lm.generate_tokens(
                            current_input,
                            max_new_tokens=1,
                            temperature=temperature,
                            do_sample=do_sample,
                            **kwargs,
                        )

                        if fallback_tokens.numel() > 0:
                            generated_tokens.extend(fallback_tokens[0].cpu().tolist())
                            current_input = torch.cat(
                                [current_input, fallback_tokens], dim=1
                            )

                        self.metrics["total_generation_time_ms"] += (
                            time.time() - step_start
                        ) * 1000

                        self.logger.info(
                            f"Step {step}: fallback generated 1 token, "
                            f"total: {len(generated_tokens)}/{max_tokens}"
                        )

                # Phase 3D: structured per-step record (best-effort)
                try:
                    if getattr(self.structured_profiler, "enable_profiling", False):
                        self.structured_profiler.record_step(
                            step=step,
                            draft_time_ms=draft_time_ms,
                            verify_time_ms=verify_time_ms,
                            acceptance_time_ms=0.0,
                            kv_append_time_ms=self.metrics.get(
                                "kv_append_time_ms", 0.0
                            ),
                            accepted_len=accepted_len,
                            proposed_len=int(draft_tokens.shape[1]),
                        )
                except Exception:
                    pass

                # Check for early stopping
                if len(generated_tokens) >= max_tokens:
                    self.logger.debug(
                        f"Early stopping: {len(generated_tokens)} >= {max_tokens}"
                    )
                    break

                # Check for EOS token
                base_info = self.base_lm.get_tokenizer_info()
                if (
                    generated_tokens
                    and generated_tokens[-1] == base_info["eos_token_id"]
                ):
                    self.logger.info("EOS token generated, stopping")
                    break

                step_time = (time.time() - step_start) * 1000
                self.logger.debug(f"Step {step} completed in {step_time:.2f}ms")

            # Finalize results
            self.metrics["total_steps"] = step
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["total_generation_time_ms"] += total_time_ms

            # Validate tokens were generated
            if not generated_tokens or len(generated_tokens) == 0:
                self.logger.warning(
                    f"No tokens generated after {step} steps! "
                    f"max_tokens={max_tokens}, proposed={self.metrics['total_proposed']}, "
                    f"accepted={self.metrics['total_accepted']}"
                )
                # Warning is logged via logger, print only if debug enabled
                if os.getenv("SPECDEC_DEBUG_PRINTS", "0").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    print(
                        f"[WARNING] No tokens generated for prompt after {step} steps",
                        flush=True,
                    )

            # Decode generated text
            if isinstance(generated_tokens, list):
                generated_text = self.base_lm.decode(torch.tensor([generated_tokens]))
            else:
                generated_text = self.base_lm.decode(generated_tokens)

            # Calculate final metrics
            acceptance_rate = (
                self.metrics["total_accepted"] / self.metrics["total_proposed"]
                if self.metrics["total_proposed"] > 0
                else 0.0
            )
            tokens_per_sec = (
                len(generated_tokens) / (total_time_ms / 1000)
                if total_time_ms > 0
                else 0.0
            )

            # Get memory usage
            process = psutil.Process()
            mem_rss_mb = process.memory_info().rss / 1024 / 1024

            # Get CUDA memory if available
            cuda_mem_allocated = 0.0
            cuda_mem_peak = 0.0
            if self.device == "cuda" and torch.cuda.is_available():
                cuda_mem_allocated = float(
                    torch.cuda.memory_allocated() / 1024 / 1024
                )  # MB
                cuda_mem_peak = float(
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )  # MB

            # Get actual dtype from optimization manager if available
            actual_dtype = str(self.dtype).replace("torch.", "")
            actual_amp_enabled = self.amp_enabled
            if self.optimization_manager:
                opt_info = self.optimization_manager.get_optimization_info()
                actual_dtype = opt_info.get("dtype", actual_dtype)
                actual_amp_enabled = opt_info.get("enabled", actual_amp_enabled)

            # Get kernel backend info
            try:
                from kernels import get_kernel_info

                kernel_info = get_kernel_info()
                kv_append_backend = kernel_info.get("kv_append_backend", "unknown")
            except ImportError:
                kv_append_backend = "unavailable"

            result = {
                "text": generated_text,
                "generated_tokens": generated_tokens,
                "latency_ms": total_time_ms,
                "proposed": self.metrics["total_proposed"],
                "accepted": self.metrics["total_accepted"],
                "acceptance_rate": acceptance_rate,
                "tokens_per_sec": tokens_per_sec,
                "steps": step,
                "verification_time_ms": self.metrics["total_verification_time_ms"],
                "generation_time_ms": self.metrics["total_generation_time_ms"],
                "kv_appended_tokens_total": self.metrics["kv_appended_tokens_total"],
                "kv_append_time_ms": self.metrics["kv_append_time_ms"],
                "kv_append_enabled": (
                    hasattr(self.base_lm, "supports_kv_append")
                    and self.base_lm.supports_kv_append()
                ),
                "kv_append_backend": kv_append_backend,
                "mem_rss_mb": mem_rss_mb,
                "cuda_mem_allocated_mb": cuda_mem_allocated,
                "cuda_mem_peak_mb": cuda_mem_peak,
                "policy": self.policy.get_info(),
                "controller": self.controller.get_info(),
                "impl": self.implementation,
                "device": self.device,
                "dtype": actual_dtype,
                "amp_enabled": actual_amp_enabled,
                "base_model": self.config["base_model"],
                "draft_model": self.config["draft_model"],
                "draft_mode": self.config.get("draft_mode", "vanilla"),
            }

            # Add profiling data if available
            if self.profiler:
                memory_stats = self.profiler.stop_memory_tracking()
                result["profiling"] = {
                    "memory_stats": memory_stats,
                    "timing_stats": self.profiler.get_timing_stats(),
                }

            # Add optimization info if available
            if self.optimization_manager:
                result["optimization"] = (
                    self.optimization_manager.get_optimization_report()
                )

            self.logger.info(
                f"Speculative decoding completed: {len(generated_tokens)} tokens "
                f"in {total_time_ms:.2f}ms "
                f"({tokens_per_sec:.2f} tokens/sec, "
                f"acceptance_rate={acceptance_rate:.3f})"
            )

            # Cleanup device cache if available
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            self.logger.error(f"Speculative decoding failed: {e}")
            raise

    def _generate_batch_baseline(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Baseline generation path: standard autoregressive decoding using only base model.

        This method runs non-speculative generation for all prompts and returns
        results in the same format as the speculative path for compatibility.

        Args:
            prompts: List of input prompt strings
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            List of result dictionaries, one per prompt
        """
        batch_size = len(prompts)
        self.logger.info(
            f"[BASELINE] Running non-speculative generation for {batch_size} prompts"
        )

        # Track memory before batch
        mem_before = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )

        # Tokenize all prompts
        tokenizer = (
            self.base_lm._tokenizer if hasattr(self.base_lm, "_tokenizer") else None
        )
        if tokenizer is None:
            raise ValueError("Base model must have a tokenizer for baseline mode")

        encoded = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            batch_input_ids = (
                encoded["input_ids"].pin_memory().to(self.device, non_blocking=True)
            )
            batch_attention_mask = (
                encoded["attention_mask"]
                .pin_memory()
                .to(self.device, non_blocking=True)
            )
        else:
            batch_input_ids = encoded["input_ids"].to(self.device)
            batch_attention_mask = encoded["attention_mask"].to(self.device)

        # Track time
        batch_start_time = time.time()

        # Generate using base model only (standard autoregressive)
        with torch.no_grad():
            outputs = self.base_lm._model.generate(  # type: ignore
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,  # type: ignore
                eos_token_id=tokenizer.eos_token_id,  # type: ignore
                return_dict_in_generate=True,
                **kwargs,
            )

        # GPU sync if CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end_time = time.time()
        total_time_ms = (batch_end_time - batch_start_time) * 1000

        # Extract generated tokens for each prompt
        batch_generated_tokens = []
        initial_lengths = batch_input_ids.shape[1]

        for i in range(batch_size):
            # Extract generated tokens (excluding prompt)
            generated_ids = outputs.sequences[i, initial_lengths:]  # type: ignore
            # Convert to list and filter out padding/EOS
            generated_list = generated_ids.cpu().tolist()
            # Filter out EOS tokens that appear after generation
            eos_token_id = tokenizer.eos_token_id  # type: ignore
            filtered_tokens = []
            for tok in generated_list:
                if tok == eos_token_id:
                    break
                filtered_tokens.append(tok)
            batch_generated_tokens.append(filtered_tokens)

        # Get kernel backend info for compatibility
        try:
            from kernels import get_kernel_info

            kernel_info = get_kernel_info()
            kv_append_backend = kernel_info.get("kv_append_backend", "unknown")
        except ImportError:
            kv_append_backend = "unavailable"

        # Build result dictionaries compatible with speculative path
        results = []
        for i, (prompt, generated_tokens) in enumerate(
            zip(prompts, batch_generated_tokens)
        ):
            # Decode generated tokens
            if generated_tokens:
                generated_text = self.base_lm.decode(generated_tokens)
            else:
                generated_text = ""

            # Calculate metrics
            num_tokens = len(generated_tokens)
            prompt_throughput = (
                num_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0
            )
            prompt_latency_ms = total_time_ms / num_tokens if num_tokens > 0 else 0.0

            # For baseline mode, all tokens are "proposed" and "accepted"
            # (compatibility with speculative path metrics)
            results.append(
                {
                    "prompt": prompt,
                    "text": generated_text,
                    "generated_text": generated_text,
                    "generated_tokens": generated_tokens,
                    "num_generated": num_tokens,
                    "batch_index": i,
                    "batch_size": batch_size,
                    "latency_ms": prompt_latency_ms,
                    "total_time_ms": total_time_ms,
                    "tokens_per_sec": prompt_throughput,
                    "throughput_tokens_per_sec": prompt_throughput,
                    "acceptance_rate": 1.0,  # All tokens accepted in baseline
                    "proposed": num_tokens,  # All generated tokens
                    "accepted": num_tokens,  # All generated tokens
                    "draft_avg_ms": 0.0,  # No draft model
                    "verify_avg_ms": 0.0,  # No verification
                    "batch_metrics": {
                        "total_steps": 1,  # Single generation step
                        "total_proposed": num_tokens,
                        "total_accepted": num_tokens,
                        "total_draft_time_ms": 0.0,
                        "total_verification_time_ms": 0.0,
                        "total_generation_time_ms": total_time_ms,
                    },
                    "kv_append_enabled": False,
                    "kv_append_backend": kv_append_backend,
                    "kv_appended_tokens": 0,
                    "kv_append_time_ms": 0.0,
                }
            )

        # Track memory after batch
        mem_after = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )
        mem_used_mb = (mem_after - mem_before) / (1024 * 1024)

        if mem_after > 0:
            print(
                f"[BASELINE] GPU memory after: {mem_after / (1024**2):.2f} MB | "
                f"Delta: {mem_used_mb:.2f} MB",
                flush=True,
            )

        self.logger.info(
            f"Baseline batch generation complete: {batch_size} prompts, "
            f"GPU memory used: {mem_used_mb:.2f} MB"
        )

        return results

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts in parallel (batched processing).

        This method processes multiple prompts simultaneously to maximize GPU utilization.
        Prompts are tokenized together, padded to the same length, and processed as a batch.

        Note: For better error diagnostics, set CUDA_LAUNCH_BLOCKING=1 environment variable.
        This forces synchronous CUDA execution and provides more accurate stack traces.

        Args:
            prompts: List of input prompt strings
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            List of result dictionaries, one per prompt
        """
        # Check if CUDA_LAUNCH_BLOCKING is set (recommended for debugging)
        if self.device == "cuda" and os.getenv("CUDA_LAUNCH_BLOCKING") == "1":
            self.logger.info(
                "CUDA_LAUNCH_BLOCKING=1 is set - using synchronous CUDA execution "
                "for better error diagnostics"
            )

        if not prompts:
            return []

        # CRITICAL: Reset KV cache manager state at the start of each batch
        # This prevents state leakage from previous batches in K-sweep benchmarks
        self.kv_cache_manager.reset()
        if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
            self.logger.debug("[BATCH] Reset KV cache manager state for fresh batch")

        batch_size = len(prompts)
        self.logger.info(f"Starting batched generation for {batch_size} prompts")
        if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
            self.logger.debug(
                f"[BATCH] Starting batch processing: {batch_size} prompts, max_tokens={max_tokens}"
            )

        # Use provided parameters or fall back to config
        max_tokens = max_tokens or self.config["max_new_tokens"]
        temperature = temperature or self.config["temperature"]
        do_sample = do_sample if do_sample is not None else self.config["do_sample"]

        # Baseline mode: non-speculative generation using only base model
        if not self.speculative_enabled:
            return self._generate_batch_baseline(
                prompts, max_tokens, temperature, do_sample, **kwargs
            )

        # Model init diagnostics (only print once, gated with SPECDEC_DEBUG)
        if not hasattr(self, "_batch_init_printed"):
            if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                self.logger.debug(
                    f"[BATCH] Base model: {self.config.get('base_model', 'unknown')}, "
                    f"Draft model: {self.config.get('draft_model', 'unknown')}, "
                    f"Device: {self.device}, Dtype: {self.dtype}"
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    self.logger.debug(f"[BATCH] GPU: {torch.cuda.get_device_name(0)}")
            self._batch_init_printed = True

        # Tokenize all prompts together with padding
        tokenizer = (
            self.base_lm._tokenizer if hasattr(self.base_lm, "_tokenizer") else None
        )
        if tokenizer is None:
            # Fallback: process sequentially if no tokenizer access
            self.logger.warning(
                "No tokenizer access, falling back to sequential processing"
            )
            if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                self.logger.debug(
                    "[BATCH] Warning: No tokenizer access, falling back to sequential"
                )
            return [
                self.generate(prompt, max_tokens, temperature, do_sample, **kwargs)
                for prompt in prompts
            ]

        # Tokenizer alignment check - ensure both models use same tokenizer
        # (only if draft model exists)
        if (
            self.speculative_enabled
            and hasattr(self.draft_lm, "_tokenizer")
            and self.draft_lm._tokenizer is not None
        ):
            draft_tokenizer = self.draft_lm._tokenizer
            if tokenizer is not None:
                # Check vocab size alignment
                base_vocab_size = getattr(tokenizer, "vocab_size", None)
                draft_vocab_size = getattr(draft_tokenizer, "vocab_size", None)
                if base_vocab_size is not None and draft_vocab_size is not None:
                    if base_vocab_size != draft_vocab_size:
                        self.logger.warning(
                            f"Tokenizer vocab size mismatch: base={base_vocab_size}, draft={draft_vocab_size}"
                        )
                    else:
                        if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                            "1",
                            "true",
                            "yes",
                        ):
                            self.logger.debug(
                                f"[CHECK] Tokenizer alignment OK - vocab_size={base_vocab_size}"
                            )

                        # One-time warmup check: tokenizer overlap sanity test
                        if not hasattr(self, "_tokenizer_warmup_done"):
                            test_text = "The quick brown fox"
                            test_text_extended = "The quick brown fox jumps"
                            try:
                                tokens_base = tokenizer(test_text, return_tensors="pt")[
                                    "input_ids"
                                ]
                                tokens_extended = tokenizer(
                                    test_text_extended, return_tensors="pt"
                                )["input_ids"]
                                # Compare overlapping prefix
                                min_len = min(
                                    tokens_base.shape[1], tokens_extended.shape[1]
                                )
                                if min_len > 0:
                                    overlap = (
                                        (
                                            tokens_base[0, :min_len]
                                            == tokens_extended[0, :min_len]
                                        )
                                        .float()
                                        .mean()
                                        .item()
                                    )
                                    if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                        "1",
                                        "true",
                                        "yes",
                                    ):
                                        self.logger.debug(
                                            f"[CHECK] Tokenizer overlap sanity: {overlap:.2f} "
                                            f"(expected ~1.0 for prefix match)"
                                        )
                                self._tokenizer_warmup_done = True
                            except Exception as e:
                                if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                ):
                                    self.logger.debug(
                                        f"[CHECK] Tokenizer warmup test failed: {e}"
                                    )

        # Tokenize with padding (pre-tokenize once, reuse across iterations)
        if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
            self.logger.debug(
                f"[BATCH] Tokenizing {batch_size} prompts with padding..."
            )
        encoded = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Use pinned memory for faster CPU->GPU transfer (if CUDA)
        if self.device == "cuda" and torch.cuda.is_available():
            # Allocate tensor with pinned memory for faster transfer
            batch_input_ids = (
                encoded["input_ids"].pin_memory().to(self.device, non_blocking=True)
            )
        else:
            batch_input_ids = encoded["input_ids"].to(self.device)
        if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
            pinned = (
                "yes" if self.device == "cuda" and torch.cuda.is_available() else "no"
            )
            self.logger.debug(
                f"[BATCH] Tokenized batch shape: {batch_input_ids.shape} (pinned={pinned})"
            )

        # Track memory before batch
        mem_before = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )
        if mem_before > 0 and os.getenv("SPECDEC_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
        ):
            self.logger.debug(
                f"[BATCH] GPU memory before: {mem_before / (1024**2):.2f} MB"
            )

        # VECTORIZED BATCH PROCESSING: Process all prompts together in parallel
        # This enables true GPU parallelism with batched tensor operations
        if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
            self.logger.debug(
                f"[BATCH] Starting vectorized speculative decoding for {batch_size} prompts"
            )

        # Pre-tokenize all prompts once (already done above)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Use optimization context
        if self.optimization_manager:
            optimization_context = self.optimization_manager.get_optimization_context()
        else:
            optimization_context = (
                amp_context(self.device, self.dtype)
                if self.amp_enabled
                else torch.no_grad()
            )

        # Clear KV caches for batch
        # NOTE: KV cache updates are disabled for batched operations due to complexity
        # of tracking per-prompt sequence states. Each prompt in a batch has independent
        # generation sequences, making shared KV cache management non-trivial.
        # Future optimization: implement per-prompt KV cache tracking for batched mode.
        if hasattr(self.base_lm, "clear_kv_cache"):
            self.base_lm.clear_kv_cache()
        if hasattr(self.draft_lm, "clear_kv_cache"):
            self.draft_lm.clear_kv_cache()

        # Check if KV cache append is enabled via environment variable
        kv_cache_enabled_env = os.getenv("SPECDEC_ENABLE_KV_APPEND", "0") == "1"
        kv_cache_supported = (
            hasattr(self.base_lm, "supports_kv_append")
            and self.base_lm.supports_kv_append()
        )
        kv_cache_enabled = kv_cache_enabled_env and kv_cache_supported

        # KV cache now managed by SafeKVCacheManager - no manual dictionaries
        if kv_cache_enabled:
            if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                self.logger.debug(
                    "[BATCH] KV cache append enabled - using SafeKVCacheManager"
                )
        else:
            if kv_cache_enabled_env and not kv_cache_supported:
                self.logger.warning(
                    "[BATCH] SPECDEC_ENABLE_KV_APPEND=1 but model doesn't support KV append"
                )

        # Keep KV manager batch metadata in sync with the current workload
        self.kv_cache_manager.set_batch_size(batch_size)

        # Configuration flags for performance and stability
        enable_debug_prints = os.getenv("SPECDEC_DEBUG_PRINTS", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        enable_periodic_sync = os.getenv("SPECDEC_PERIODIC_SYNC", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        enable_periodic_cache_clear = os.getenv(
            "SPECDEC_PERIODIC_CACHE_CLEAR", "0"
        ).lower() in ("1", "true", "yes")

        # Initialize sequence pool if enabled
        enable_sequence_pool = os.getenv(
            "SPECDEC_ENABLE_SEQUENCE_POOL", "0"
        ).lower() in ("1", "true", "yes")
        max_group_size = None
        min_group_size = 1
        if enable_sequence_pool:
            max_group_size_str = os.getenv("SPECDEC_SEQUENCE_POOL_MAX_GROUP_SIZE", "")
            if max_group_size_str:
                try:
                    max_group_size = int(max_group_size_str)
                except ValueError:
                    max_group_size = None

            min_group_size_str = os.getenv("SPECDEC_SEQUENCE_POOL_MIN_GROUP_SIZE", "1")
            try:
                min_group_size = int(min_group_size_str)
            except ValueError:
                min_group_size = 1

            if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                self.logger.debug(
                    f"[SEQUENCE_POOL] Enabled: max_group_size={max_group_size}, "
                    f"min_group_size={min_group_size}"
                )

        sequence_pool = (
            SequencePool(
                max_group_size=max_group_size,
                min_group_size=min_group_size,
            )
            if enable_sequence_pool
            else None
        )

        # Reset metrics for batch
        batch_metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_generated_tokens": 0,
            "total_steps": 0,
            "total_draft_time_ms": 0.0,
            "total_verification_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
        }

        with optimization_context:
            # Vectorized speculative decoding loop
            # All prompts processed together in batched tensor operations
            # Store sequences as list of 1D tensors to handle variable lengths
            current_input_ids = [
                batch_input_ids[i].clone() for i in range(batch_size)
            ]  # List of [seq_len] tensors
            batch_generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
            batch_active = [
                True
            ] * batch_size  # Track which prompts are still generating
            # Track per-prompt proposed/accepted for accurate metrics
            per_prompt_proposed_counts = [0] * batch_size
            per_prompt_accepted_counts = [0] * batch_size

            # Track global sequence IDs and lengths for KV cache alignment
            # Global sequence IDs are just 0 to batch_size-1 (one per prompt)
            global_sequence_ids = list(range(batch_size))

            # Initialize sequence lengths from initial tokenization
            initial_sequence_lengths = [seq.shape[0] for seq in current_input_ids]

            # ZERO-COPY POINTER ROLLBACK: Track current_seq_lens as source of truth
            # These pointers track the valid length in pre-allocated buffers
            # When tokens are rejected, we simply decrement the pointer (fast rewind)
            current_seq_lens = (
                initial_sequence_lengths.copy()
            )  # Per-sequence source of truth

            if kv_cache_enabled:
                self.kv_cache_manager.set_sequence_metadata(
                    global_sequence_ids=global_sequence_ids,
                    sequence_lengths=initial_sequence_lengths,
                )
                # Initialize cache manager's current_seq_lens to match initial lengths
                if len(self.kv_cache_manager.base_current_seq_lens) != batch_size:
                    self.kv_cache_manager.base_current_seq_lens = (
                        initial_sequence_lengths.copy()
                    )
                if len(self.kv_cache_manager.draft_current_seq_lens) != batch_size:
                    self.kv_cache_manager.draft_current_seq_lens = (
                        initial_sequence_lengths.copy()
                    )

            # Initialize sequence pool with all sequences
            if sequence_pool is not None:
                for global_id, seq in zip(global_sequence_ids, current_input_ids):
                    sequence_pool.add_sequence(global_id, seq, is_active=True)

            step = 0
            generation_start = time.time()

            # Create CUDA streams for draft/verify overlap
            # CRITICAL: Disable streams if CUDA_LAUNCH_BLOCKING=1 (forces synchronous execution)
            # Streams won't provide benefit and may cause confusion with synchronous mode
            draft_stream = None
            verify_stream = None
            cuda_launch_blocking = os.getenv("CUDA_LAUNCH_BLOCKING") == "1"

            if (
                self.device == "cuda"
                and torch.cuda.is_available()
                and not cuda_launch_blocking
            ):
                # Use scheduler's streams if available, otherwise create new ones
                if (
                    hasattr(self.scheduler, "verification_stream")
                    and self.scheduler.verification_stream
                ):
                    verify_stream = self.scheduler.verification_stream
                    draft_stream = torch.cuda.current_stream()
                else:
                    draft_stream = torch.cuda.Stream()
                    verify_stream = torch.cuda.Stream()
            elif cuda_launch_blocking:
                self.logger.info(
                    "CUDA_LAUNCH_BLOCKING=1 detected - disabling async streams "
                    "for synchronous execution (better for debugging)"
                )

            while step < max_tokens:
                step += 1
                active_count = sum(batch_active)

                # Debug print when active_count drops (gated with SPECDEC_DEBUG)
                if step > 1 and active_count < batch_size:
                    finished_count = batch_size - active_count
                    if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                        self.logger.debug(
                            f"[INFO] Step {step}: {finished_count} sequence(s) finished, {active_count} still active"
                        )

                if active_count == 0:
                    if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                        self.logger.debug(
                            f"[INFO] Step {step}: All sequences finished, exiting loop"
                        )
                    break  # All prompts finished

                # Get K from controller (same K for all active prompts)
                context = {
                    "step": step,
                    "generated_tokens": max(
                        len(tokens) for tokens in batch_generated_tokens
                    ),
                    "acceptance_rate": (
                        batch_metrics["total_accepted"]
                        / max(batch_metrics["total_proposed"], 1)
                    ),
                }
                k = self.controller.get_k(step, context)

                # Validate K - must be > 0 for generation
                if k <= 0:
                    self.logger.warning(f"Step {step}: K={k} <= 0, skipping generation")
                    break

                # Filter to active prompts only
                active_indices = [i for i, active in enumerate(batch_active) if active]
                if not active_indices:
                    break

                # Update sequence pool with current sequences if enabled
                if sequence_pool is not None:
                    for i in active_indices:
                        global_id = global_sequence_ids[i]
                        seq = current_input_ids[i]
                        is_active = batch_active[i]
                        sequence_pool.add_sequence(global_id, seq, is_active=is_active)

                # Get batches from pool if enabled, otherwise use all active sequences
                if sequence_pool is not None:
                    # Process same-length groups first, then mixed batches
                    batches_to_process = []

                    # Get same-length groups until none available
                    while True:
                        same_length_batch = sequence_pool.get_same_length_group()
                        if same_length_batch is None:
                            break
                        global_ids, sequences, length = same_length_batch
                        batches_to_process.append((global_ids, sequences, length, True))

                    # Get mixed-length batch for remaining sequences
                    mixed_batch = sequence_pool.get_mixed_length_batch()
                    if mixed_batch is not None:
                        global_ids, sequences = mixed_batch
                        batches_to_process.append((global_ids, sequences, None, False))

                    if not batches_to_process:
                        break

                    # Process each batch (for now, process first batch only to maintain structure)
                    # TODO: Refactor to process all batches per step
                    if batches_to_process:
                        (
                            batch_global_ids,
                            batch_sequences,
                            batch_length,
                            batch_is_same_length,
                        ) = batches_to_process[0]
                        # Map global IDs back to active indices
                        global_to_active = {
                            gid: i for i, gid in enumerate(global_sequence_ids)
                        }
                        active_indices = [
                            global_to_active[gid]
                            for gid in batch_global_ids
                            if gid in global_to_active
                        ]
                        active_seqs = [
                            seq.detach().clone().contiguous() for seq in batch_sequences
                        ]
                        _is_same_length_batch = batch_is_same_length
                    else:
                        break
                else:
                    # Original behavior: use all active sequences
                    active_seqs = [
                        current_input_ids[i].detach().clone().contiguous()
                        for i in active_indices
                    ]
                    _is_same_length_batch = False

                if len(active_seqs) == 0:
                    break

                # Update KV cache manager with active indices and sequence metadata
                # Use relative indices (0 to len-1) for KV cache, not absolute batch indices
                relative_indices = list(range(len(active_indices)))
                self.kv_cache_manager.set_active_indices(relative_indices)
                relative_indices_tensor = torch.tensor(
                    relative_indices, dtype=torch.long, device=self.device
                )

                # Get global sequence IDs and current lengths for active sequences
                active_global_ids = [global_sequence_ids[i] for i in active_indices]
                active_sequence_lengths = [
                    current_input_ids[i].shape[0] for i in active_indices
                ]

                # Update KV cache manager with current sequence metadata
                if kv_cache_enabled:
                    self.kv_cache_manager.set_sequence_metadata(
                        global_sequence_ids=active_global_ids,
                        sequence_lengths=active_sequence_lengths,
                        batch_to_global_map={
                            i: global_id
                            for i, global_id in enumerate(active_global_ids)
                        },
                    )

                # CRITICAL: Validate immediately after extraction to catch corruption early
                # Use GPU-only validation to avoid CPU-GPU sync overhead
                # Only sync to CPU if corruption is actually detected
                draft_vocab_size_check = get_vocab_size(self.draft_lm)
                if draft_vocab_size_check is not None:
                    for i, seq in enumerate(active_seqs):
                        # GPU-only validation: check without CPU sync (faster)
                        invalid_mask = (seq >= draft_vocab_size_check) | (seq < 0)
                        if invalid_mask.any():
                            # Only sync to CPU if corruption detected (rare case)
                            min_val = torch.min(seq).item()
                            max_val = torch.max(seq).item()
                            raise RuntimeError(
                                f"CRITICAL: Corrupted sequence detected at step {step}, "
                                f"prompt {active_indices[i]}: "
                                f"Min={min_val}, Max={max_val}, Vocab_size={draft_vocab_size_check}"
                            )

                # CRITICAL: Validate sequences with DRAFT vocab size since we're passing to draft model
                # This is the root cause fix - draft model may have different vocab size than base
                draft_vocab_size = get_vocab_size(self.draft_lm)
                base_vocab_size = get_vocab_size(self.base_lm)

                # Use minimum vocab size to ensure tokens are valid for both models
                vocab_size = None
                if draft_vocab_size is not None and base_vocab_size is not None:
                    vocab_size = min(draft_vocab_size, base_vocab_size)
                elif draft_vocab_size is not None:
                    vocab_size = draft_vocab_size
                elif base_vocab_size is not None:
                    vocab_size = base_vocab_size

                # Validate sequences with appropriate vocab size BEFORE padding
                if vocab_size is not None and len(active_seqs) > 0:
                    # Validate each sequence in-place on GPU (minimal CPU overhead)
                    # This ensures all tokens are valid before padding
                    for i, seq in enumerate(active_seqs):
                        active_seqs[i] = validate_and_clamp_tokens(
                            seq, vocab_size, f"seq_{i}"
                        )

                # Pad sequences using utility function
                # CRITICAL: Ensure pad_value is valid for both base and draft models
                pad_value = (
                    tokenizer.pad_token_id
                    if tokenizer is not None and tokenizer.pad_token_id is not None
                    else 0
                )
                # Validate pad_value against the vocab size we're using
                if vocab_size is not None and (
                    pad_value >= vocab_size or pad_value < 0
                ):
                    pad_value = 0  # Use 0 as safe default (always valid for GPT models)

                # Use sequence utility to pad sequences
                # For same-length groups from pool, no padding needed
                if sequence_pool is not None and _is_same_length_batch:
                    # All sequences have same length, just stack them
                    active_input_ids = torch.stack(active_seqs, dim=0).contiguous()
                    seq_length = active_seqs[0].shape[0]
                    active_attention_mask = torch.ones(
                        (len(active_seqs), seq_length),
                        dtype=torch.long,
                        device=self.device,
                    )
                    original_lengths = [seq_length] * len(active_seqs)
                else:
                    # Mixed lengths or pool disabled: pad as usual
                    active_input_ids, active_attention_mask, original_lengths = (
                        pad_sequences(
                            sequences=active_seqs,
                            pad_token_id=pad_value,
                            device=torch.device(self.device),
                        )
                    )

                # CRITICAL: Final validation with draft vocab size before passing to draft model
                # This is the last safety check before embedding layer
                draft_vocab_size = get_vocab_size(self.draft_lm)
                if draft_vocab_size is not None:
                    active_input_ids = validate_and_clamp_tokens(
                        active_input_ids, draft_vocab_size, "batch_input_draft"
                    )

                # Create explicit position IDs starting from 0 for each sequence
                max_seq_len = (
                    active_input_ids.shape[1] if active_input_ids.shape[0] > 0 else 0
                )
                active_position_ids = create_position_ids(
                    sequence_lengths=original_lengths,
                    max_length=max_seq_len,
                    device=torch.device(self.device),
                )

                # CRITICAL: Always compute non_pad_tokens for padding check
                # This check is required for correctness, not just debugging
                non_pad_tokens_tensor = (active_input_ids != pad_value).sum()
                total_tokens = active_input_ids.numel()
                # Only sync to CPU if debug logging is enabled (optimization)
                if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                    non_pad_tokens = non_pad_tokens_tensor.item()
                else:
                    # For the check, we can use tensor comparison directly (no CPU sync)
                    # But we need to handle both tensor and scalar cases
                    non_pad_tokens = (
                        non_pad_tokens_tensor  # Keep as tensor for GPU comparison
                    )

                # Debug instrumentation for batch input (gated by SPECDEC_DEBUG_BATCH_INPUT)
                if step == 1 and os.getenv(
                    "SPECDEC_DEBUG_BATCH_INPUT", "0"
                ).lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    non_pad_count = (
                        non_pad_tokens.item()
                        if isinstance(non_pad_tokens, torch.Tensor)
                        else non_pad_tokens
                    )
                    self.logger.info(
                        f"[BATCH_INPUT_DEBUG] Step 1 - active_input_ids shape: {active_input_ids.shape}, "
                        f"active={len(active_indices)}, max_len={max_seq_len}, "
                        f"original_lens={original_lengths}, "
                        f"non_pad={non_pad_count}/{total_tokens}"
                    )
                    # Decode first two sequences to verify prompts are present
                    if tokenizer is not None and active_input_ids.shape[0] > 0:
                        for seq_idx in range(min(2, active_input_ids.shape[0])):
                            seq_tokens = active_input_ids[seq_idx]
                            # Remove padding before decoding
                            seq_non_pad = seq_tokens[seq_tokens != pad_value]
                            if len(seq_non_pad) > 0:
                                try:
                                    decoded_text = tokenizer.decode(seq_non_pad)
                                    self.logger.info(
                                        f"[BATCH_INPUT_DEBUG] Sequence {seq_idx} decoded: {decoded_text[:100]}..."
                                    )
                                except Exception as e:
                                    self.logger.warning(
                                        f"[BATCH_INPUT_DEBUG] Failed to decode sequence {seq_idx}: {e}"
                                    )
                            else:
                                self.logger.warning(
                                    f"[BATCH_INPUT_DEBUG] Sequence {seq_idx} has no non-padding tokens!"
                                )

                # Also log in regular debug mode (less verbose)
                if step == 1 and os.getenv("SPECDEC_DEBUG", "0").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    non_pad_count = (
                        non_pad_tokens.item()
                        if isinstance(non_pad_tokens, torch.Tensor)
                        else non_pad_tokens
                    )
                    self.logger.debug(
                        f"[BATCH] Batched active_input_ids shape: {active_input_ids.shape} "
                        f"(active={len(active_indices)}, max_len={max_seq_len}, "
                        f"original_lens={original_lengths}, "
                        f"non_pad={non_pad_count}/{total_tokens})"
                    )

                # Check if all tokens are padding (GPU-only comparison, no CPU sync)
                # Use tensor comparison to avoid CPU sync when debug is disabled
                if isinstance(non_pad_tokens, torch.Tensor):
                    # GPU tensor comparison (no CPU sync)
                    if non_pad_tokens.item() == 0:
                        self.logger.warning(
                            f"Step {step}: All tokens are padding, skipping generation"
                        )
                        break
                elif non_pad_tokens == 0:
                    # CPU scalar (when debug enabled and already synced)
                    self.logger.warning(
                        f"Step {step}: All tokens are padding, skipping generation"
                    )
                    break

                # Step 1: Generate draft tokens in batch (on draft stream)
                draft_start_event = None
                draft_end_event = None
                if (
                    self.device == "cuda"
                    and torch.cuda.is_available()
                    and draft_stream is not None
                ):
                    draft_start_event = torch.cuda.Event(enable_timing=True)
                    draft_end_event = torch.cuda.Event(enable_timing=True)
                    draft_start_event.record(draft_stream)

                # Use CUDA events for accurate timing (same as verify)
                draft_start_wall = time.time()

                # Debug: log before draft execution (only first step, gated with SPECDEC_DEBUG)
                if step == 1 and os.getenv("SPECDEC_DEBUG", "0").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    self.logger.debug(
                        f"[DEBUG] Before draft - input shape: {active_input_ids.shape}, "
                        f"K={k}, device={active_input_ids.device}"
                    )

                # Apply temperature stabilization for draft model
                # Use lower effective temperature to improve acceptance rate
                draft_temperature = (
                    max(temperature / 1.5, 0.1) if temperature is not None else 0.7
                )

                # Prepare past_key_values for draft model using centralized KV cache manager
                # ZERO-COPY: Pass current_seq_lens to model for proper attention masking
                draft_past_kv = None
                draft_current_seq_lens_for_model = None
                if kv_cache_enabled:
                    # Use relative indices for KV cache retrieval (0 to active_count-1)
                    draft_past_kv = self.kv_cache_manager.get_draft_past_kv(
                        relative_indices_tensor
                    )
                    # Extract current_seq_lens for active sequences (for attention mask construction)
                    draft_current_seq_lens_for_model = [
                        (
                            self.kv_cache_manager.draft_current_seq_lens[i]
                            if i < len(self.kv_cache_manager.draft_current_seq_lens)
                            else current_seq_lens[active_indices[i]]
                        )
                        for i in range(len(active_indices))
                    ]

                # OPTIMAL FIX: Prepare tensor ONCE before both streams (best practice)
                # Root cause: active_input_ids can be corrupted if shared between streams
                # Solution: Clone once synchronously, then both streams use independent copies
                # Performance: Minimal overhead (single clone) vs massive speedup (parallel execution)
                draft_vocab_size = get_vocab_size(self.draft_lm)
                base_vocab_size = get_vocab_size(self.base_lm)

                if draft_stream is not None or verify_stream is not None:
                    # OPTIMIZATION: Stream synchronization handled by CUDA events
                    # We rely on event-based synchronization instead of full device sync
                    # This is more efficient and allows better overlap
                    # Note: CUDA events in scheduler handle the necessary synchronization

                    # OPTIMAL: Clone ONCE with detach() BEFORE streams for complete independence
                    # This breaks computation graph and ensures no shared memory between streams
                    # Performance: Single clone operation is negligible vs model execution time
                    # Both draft and verify streams will use this prepared tensor
                    try:
                        # Use detach().clone() to ensure complete independence from computation graph
                        # This prevents any shared memory or reference issues with CUDA streams
                        active_input_ids_prepared = (
                            active_input_ids.detach().clone().contiguous()
                        )

                        # CRITICAL: Validate BEFORE entering streams (synchronous validation)
                        # Validate with draft vocab size since draft is more restrictive
                        if draft_vocab_size is not None:
                            active_input_ids_prepared = validate_and_clamp_tokens(
                                active_input_ids_prepared,
                                draft_vocab_size,
                                "batch_input_pre_stream",
                                strict=True,
                            )

                            # GPU-only validation: check without CPU sync (faster)
                            # Only sync to CPU if corruption detected (rare case)
                            invalid_mask = (
                                active_input_ids_prepared >= draft_vocab_size
                            ) | (active_input_ids_prepared < 0)
                            if invalid_mask.any():
                                min_val = torch.min(active_input_ids_prepared).item()
                                max_val = torch.max(active_input_ids_prepared).item()
                                raise RuntimeError(
                                    f"CRITICAL: Invalid tokens before stream! "
                                    f"Min={min_val}, Max={max_val}, Vocab_size={draft_vocab_size}"
                                )

                    except Exception as e:
                        self.logger.error(
                            f"Step {step}: Failed to prepare tensor for streams! Error: {e}"
                        )
                        raise RuntimeError(
                            f"Failed to prepare tensor for CUDA streams: {e}"
                        ) from e

                # Draft stream: Use prepared tensor
                if draft_stream is not None:
                    with torch.cuda.stream(draft_stream):
                        try:
                            draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                                active_input_ids_prepared,
                                max_new_tokens=k,
                                temperature=draft_temperature,  # Lower temperature for draft
                                do_sample=False,  # Use greedy for draft to maximize acceptance
                                stream=draft_stream,
                                past_key_values=draft_past_kv,
                                attention_mask=active_attention_mask,  # Pass attention mask to skip padding
                                position_ids=active_position_ids,  # Pass explicit position IDs
                                **kwargs,
                            )
                        except RuntimeError as e:
                            # Catch CUDA errors and provide better diagnostics
                            if "indexSelectLargeIndex" in str(
                                e
                            ) or "device-side assert" in str(e):
                                vocab_info = (
                                    getattr(
                                        self.draft_lm._model.config,
                                        "vocab_size",
                                        "unknown",
                                    )
                                    if hasattr(self.draft_lm, "_model")
                                    else "unknown"
                                )
                                self.logger.error(
                                    f"Step {step}: CUDA embedding error in draft model! "
                                    f"active_input_ids.shape={active_input_ids.shape}, "
                                    f"vocab_size={vocab_info}, Error: {e}"
                                )
                            raise
                    if draft_end_event is not None:
                        draft_end_event.record(draft_stream)
                else:
                    # For non-stream path, still validate before model call
                    if draft_vocab_size is not None:
                        active_input_ids = validate_and_clamp_tokens(
                            active_input_ids,
                            draft_vocab_size,
                            "draft_input_sync",
                            strict=True,
                        )

                        # GPU-only validation: check without CPU sync (faster)
                        # Only sync to CPU if corruption detected (rare case)
                        invalid_mask = (active_input_ids >= draft_vocab_size) | (
                            active_input_ids < 0
                        )
                        if invalid_mask.any():
                            min_val = torch.min(active_input_ids).item()
                            max_val = torch.max(active_input_ids).item()
                            raise RuntimeError(
                                f"CRITICAL: Invalid tokens AFTER validation! "
                                f"Min={min_val}, Max={max_val}, Vocab_size={draft_vocab_size}"
                            )

                    draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                        active_input_ids,
                        max_new_tokens=k,
                        temperature=draft_temperature,  # Lower temperature for draft
                        do_sample=False,  # Use greedy for draft to maximize acceptance
                        past_key_values=draft_past_kv,
                        attention_mask=active_attention_mask,  # Pass attention mask to skip padding
                        position_ids=active_position_ids,  # Pass explicit position IDs
                        current_seq_lens=draft_current_seq_lens_for_model,  # ZERO-COPY: Pass for proper masking
                        **kwargs,
                    )

                # Update draft KV cache using centralized manager
                if kv_cache_enabled and active_count > 0:
                    # Get KV cache from draft model
                    draft_new_kv = None
                    if (
                        hasattr(self.draft_lm, "_last_generated_kv_raw")
                        and self.draft_lm._last_generated_kv_raw is not None
                    ):
                        draft_new_kv = self.draft_lm._last_generated_kv_raw
                    elif (
                        hasattr(self.draft_lm, "_last_generated_kv")
                        and self.draft_lm._last_generated_kv is not None
                    ):
                        if hasattr(self.draft_lm._last_generated_kv, "past_key_values"):
                            draft_new_kv = (
                                self.draft_lm._last_generated_kv.past_key_values
                            )
                        else:
                            draft_new_kv = self.draft_lm._last_generated_kv

                    # Update using centralized manager (handles batch dimension tracking)
                    if draft_new_kv is not None:
                        if isinstance(draft_new_kv, tuple):
                            self.kv_cache_manager.update_draft_cache(draft_new_kv)
                        else:
                            # Convert to tuple if needed
                            if len(draft_new_kv) > 0:
                                self.kv_cache_manager.update_draft_cache(
                                    tuple(draft_new_kv)
                                )

                # Validate draft outputs
                if draft_tokens.numel() == 0 or draft_tokens.shape[1] == 0:
                    self.logger.error(
                        f"Step {step}: Draft model returned empty tokens! Shape: {draft_tokens.shape}"
                    )
                    break

                # Debug: decode draft tokens (only first step, gated with SPECDEC_DEBUG)
                if step == 1 and tokenizer is not None:
                    if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                        try:
                            draft_text = tokenizer.decode(draft_tokens[0])
                            self.logger.debug(
                                f"[DEBUG] Draft tokens decoded (first prompt): {draft_text[:100]}..."
                            )
                        except Exception as e:
                            self.logger.debug(
                                f"[DEBUG] Failed to decode draft tokens: {e}"
                            )

                # Calculate draft time using CUDA events if available, otherwise wall-clock
                if draft_start_event is not None and draft_end_event is not None:
                    draft_end_event.synchronize()
                    draft_time_ms = draft_start_event.elapsed_time(draft_end_event)
                else:
                    draft_time_ms = (time.time() - draft_start_wall) * 1000

                # Debug: log draft outputs shape (reduced frequency, gated with SPECDEC_DEBUG)
                if (step == 1 or step % 16 == 0) and os.getenv(
                    "SPECDEC_DEBUG", "0"
                ).lower() in ("1", "true", "yes"):
                    self.logger.debug(
                        f"[DEBUG] Draft execution - tokens shape: {draft_tokens.shape}, "
                        f"logits shape: {draft_logits.shape}, "
                        f"time: {draft_time_ms:.2f}ms, "
                        f"proposed_tokens: {draft_tokens.shape[0] * draft_tokens.shape[1]}"
                    )
                if step == 1 and os.getenv("SPECDEC_DEBUG", "0").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    # Log sample token IDs (only first step)
                    self.logger.debug(
                        f"[DEBUG] Sample draft token IDs (first batch): "
                        f"{draft_tokens[0, :min(5, draft_tokens.shape[1])].tolist()}"
                    )

                # Step 2: Verify with base model in batch (on verify stream for true overlap)
                # Use scheduler for batched verification if available, otherwise direct call
                # Verification can start immediately on separate stream (uses same input_ids and k)
                # This enables true parallel execution of draft and verify
                verify_start_event = None
                verify_end_event = None
                if (
                    self.device == "cuda"
                    and torch.cuda.is_available()
                    and verify_stream is not None
                ):
                    verify_start_event = torch.cuda.Event(enable_timing=True)
                    verify_end_event = torch.cuda.Event(enable_timing=True)

                # Use CUDA events for accurate verify timing (same as draft)
                verify_start_wall = time.time()

                # CRITICAL: Initialize base_current_seq_lens_for_model before all code paths
                # to prevent UnboundLocalError in fallback/non-KV-cache scenarios
                base_past_kv = None
                base_current_seq_lens_for_model: Optional[List[int]] = None
                if kv_cache_enabled:
                    # Use relative indices for KV cache retrieval (0 to active_count-1)
                    base_past_kv = self.kv_cache_manager.get_base_past_kv(
                        relative_indices_tensor
                    )
                    # Extract current_seq_lens for active sequences (for attention mask construction)
                    base_current_seq_lens_for_model = [
                        (
                            self.kv_cache_manager.base_current_seq_lens[i]
                            if i < len(self.kv_cache_manager.base_current_seq_lens)
                            else current_seq_lens[active_indices[i]]
                        )
                        for i in range(len(active_indices))
                    ]

                # Use scheduler's multi-stream verification if available for proper stream management
                # Create dummy draft tokens tensor for scheduler API (it needs draft_tokens shape)
                # Note: We use actual draft_tokens shape but scheduler doesn't use the values
                dummy_draft_tokens = torch.zeros(
                    (active_input_ids.shape[0], k),
                    dtype=torch.long,
                    device=self.device,
                )

                # Try to use scheduler for verification (better stream management)
                if (
                    self.scheduler is not None
                    and self.scheduler.enable_multi_stream
                    and verify_stream is not None
                    and draft_stream is not None
                ):
                    # TRUE OVERLAP: Scheduler handles multi-stream verification
                    # Both operations run concurrently on GPU via scheduler
                    with torch.cuda.stream(verify_stream):
                        if verify_start_event is not None:
                            verify_start_event.record(verify_stream)
                        # Use scheduler's verification (it manages streams internally)
                        # Use prepared tensor for consistency and safety
                        # Note: scheduler will extract attention_mask from kwargs if present
                        base_tokens, base_logits, verify_info = (
                            self.scheduler.schedule_verification(
                                self.base_lm,
                                dummy_draft_tokens,  # Shape only, not actual draft tokens
                                active_input_ids_prepared,
                                temperature=temperature,
                                do_sample=do_sample,
                                attention_mask=active_attention_mask,  # Pass attention mask to skip padding
                                position_ids=active_position_ids,  # Pass explicit position IDs
                                **kwargs,
                            )
                        )
                        if verify_end_event is not None:
                            verify_end_event.record(verify_stream)

                    # Synchronize both streams (they may have overlapped significantly)
                    # Wait for draft to complete first (we need its outputs for acceptance)
                    if draft_end_event is not None:
                        draft_end_event.synchronize()
                    # Then wait for verify (may already be done due to overlap)
                    if verify_end_event is not None:
                        verify_end_event.synchronize()

                    # CRITICAL FIX: Sync device after event sync only if periodic sync enabled
                    # Event sync ensures streams are synchronized. Full device sync is only needed
                    # for timing accuracy or when explicitly requested to prevent unresponsiveness.
                    # For long runs, minimize syncs to reduce CPU overhead.
                    if (
                        enable_periodic_sync
                        and self.device == "cuda"
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.synchronize()

                    # Calculate verify time using CUDA events if available
                    if verify_start_event is not None and verify_end_event is not None:
                        verify_time_ms = verify_start_event.elapsed_time(
                            verify_end_event
                        )
                    else:
                        verify_time_ms = (time.time() - verify_start_wall) * 1000
                elif verify_stream is not None and draft_stream is not None:
                    # Manual stream management if scheduler not available
                    # base_past_kv and base_current_seq_lens_for_model already initialized above

                    with torch.cuda.stream(verify_stream):
                        if verify_start_event is not None:
                            verify_start_event.record(verify_stream)
                        # For verification, always use greedy (argmax) to ensure deterministic matching
                        # This ensures base_logits.argmax() matches the tokens we compare
                        try:
                            # Use prepared tensor (same as draft stream) for consistency and safety
                            base_tokens, base_logits = self.base_lm.generate_tokens(
                                active_input_ids_prepared,
                                max_new_tokens=k,  # Use k directly, not draft_tokens.shape[1]
                                temperature=1.0,  # Temperature=1.0 for deterministic argmax
                                do_sample=False,  # Always use greedy for verification
                                stream=verify_stream,
                                past_key_values=base_past_kv,
                                attention_mask=active_attention_mask,  # Pass attention mask to skip padding
                                position_ids=active_position_ids,  # Pass explicit position IDs
                                current_seq_lens=base_current_seq_lens_for_model,  # ZERO-COPY: Pass for proper masking
                                **kwargs,
                            )
                        except RuntimeError as e:
                            # Catch CUDA errors and provide better diagnostics
                            if "indexSelectLargeIndex" in str(
                                e
                            ) or "device-side assert" in str(e):
                                vocab_info = (
                                    getattr(
                                        self.base_lm._model.config,
                                        "vocab_size",
                                        "unknown",
                                    )
                                    if hasattr(self.base_lm, "_model")
                                    else "unknown"
                                )
                                self.logger.error(
                                    f"Step {step}: CUDA embedding error in base model (scheduler path)! "
                                    f"active_input_ids.shape={active_input_ids.shape}, "
                                    f"vocab_size={vocab_info}, "
                                    f"base_past_kv is None={base_past_kv is None}, Error: {e}"
                                )
                            raise
                        if verify_end_event is not None:
                            verify_end_event.record(verify_stream)

                    # Update base KV cache using centralized manager
                    if kv_cache_enabled and active_count > 0:
                        # Get KV cache from base model
                        base_new_kv = None
                        if (
                            hasattr(self.base_lm, "_last_generated_kv_raw")
                            and self.base_lm._last_generated_kv_raw is not None
                        ):
                            base_new_kv = self.base_lm._last_generated_kv_raw
                        elif (
                            hasattr(self.base_lm, "_last_generated_kv")
                            and self.base_lm._last_generated_kv is not None
                        ):
                            if hasattr(
                                self.base_lm._last_generated_kv, "past_key_values"
                            ):
                                base_new_kv = (
                                    self.base_lm._last_generated_kv.past_key_values
                                )
                            else:
                                base_new_kv = self.base_lm._last_generated_kv

                        # Update using centralized manager
                        # Note: We append KV for all K generated tokens, but will realign
                        # after acceptance to only keep accepted positions
                        if base_new_kv is not None:
                            if isinstance(base_new_kv, tuple):
                                self.kv_cache_manager.update_base_cache(
                                    base_new_kv, active_indices, tokens_appended=None
                                )
                            else:
                                if len(base_new_kv) > 0:
                                    self.kv_cache_manager.update_base_cache(
                                        tuple(base_new_kv),
                                        active_indices,
                                        tokens_appended=None,
                                    )

                    # Synchronize both streams
                    if draft_end_event is not None:
                        draft_end_event.synchronize()
                    if verify_end_event is not None:
                        verify_end_event.synchronize()

                    # CRITICAL FIX: Always sync device after event sync to ensure clean GPU state
                    # Event sync ensures streams are synchronized, but device sync ensures
                    # all operations complete before we access results. This is essential
                    # for Kaggle notebooks to prevent unresponsiveness.
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Calculate verify time using CUDA events if available
                    if verify_start_event is not None and verify_end_event is not None:
                        verify_time_ms = verify_start_event.elapsed_time(
                            verify_end_event
                        )
                    else:
                        verify_time_ms = (time.time() - verify_start_wall) * 1000
                else:
                    # Fallback: sequential execution
                    if draft_stream is not None and draft_end_event is not None:
                        draft_end_event.synchronize()
                        # OPTIMIZATION: Event sync is sufficient, no need for full device sync
                        # This reduces CPU-GPU synchronization overhead

                    # base_past_kv and base_current_seq_lens_for_model already initialized above

                    # Single validation point before model call
                    base_vocab_size = get_vocab_size(self.base_lm)
                    if base_vocab_size is not None:
                        active_input_ids = validate_and_clamp_tokens(
                            active_input_ids, base_vocab_size, "base_input"
                        )

                    # For verification, always use greedy (argmax) to ensure deterministic matching
                    base_tokens, base_logits = self.base_lm.generate_tokens(
                        active_input_ids,
                        max_new_tokens=k,
                        temperature=1.0,  # Temperature=1.0 for deterministic argmax
                        do_sample=False,  # Always use greedy for verification
                        past_key_values=base_past_kv,
                        attention_mask=active_attention_mask,  # Pass attention mask to skip padding
                        position_ids=active_position_ids,  # Pass explicit position IDs
                        current_seq_lens=base_current_seq_lens_for_model,  # ZERO-COPY: Pass for proper masking
                        **kwargs,
                    )

                    # Update base KV cache using centralized manager
                    if kv_cache_enabled:
                        base_new_kv = None
                        if (
                            hasattr(self.base_lm, "_last_generated_kv_raw")
                            and self.base_lm._last_generated_kv_raw is not None
                        ):
                            base_new_kv = self.base_lm._last_generated_kv_raw
                        elif (
                            hasattr(self.base_lm, "_last_generated_kv")
                            and self.base_lm._last_generated_kv is not None
                        ):
                            if hasattr(
                                self.base_lm._last_generated_kv, "past_key_values"
                            ):
                                base_new_kv = (
                                    self.base_lm._last_generated_kv.past_key_values
                                )
                            else:
                                base_new_kv = self.base_lm._last_generated_kv

                        if base_new_kv is not None:
                            if isinstance(base_new_kv, tuple):
                                self.kv_cache_manager.update_base_cache(
                                    base_new_kv, active_indices, tokens_appended=None
                                )
                            else:
                                if len(base_new_kv) > 0:
                                    self.kv_cache_manager.update_base_cache(
                                        tuple(base_new_kv),
                                        active_indices,
                                        tokens_appended=None,
                                    )

                    # Use wall-clock time for fallback
                    verify_time_ms = (time.time() - verify_start_wall) * 1000

                # Calculate actual overlap time using CUDA events if available
                if (
                    draft_start_event is not None
                    and draft_end_event is not None
                    and verify_start_event is not None
                    and verify_end_event is not None
                ):
                    draft_time_cuda = draft_start_event.elapsed_time(draft_end_event)
                    verify_time_cuda = verify_start_event.elapsed_time(verify_end_event)
                    # Overlap = min(draft_time, verify_time) since they run concurrently
                    # The actual time saved depends on when verify starts
                    max_time = max(draft_time_cuda, verify_time_cuda)
                    sequential_time = draft_time_cuda + verify_time_cuda
                    overlap_saved_ms = sequential_time - max_time
                    if (
                        overlap_saved_ms > 1.0 and enable_debug_prints
                    ):  # Only log if significant overlap and debug enabled
                        print(
                            f"[BATCH] Stream overlap: saved {overlap_saved_ms:.1f}ms "
                            f"(draft={draft_time_cuda:.1f}ms, verify={verify_time_cuda:.1f}ms)",
                            flush=True,
                        )

                batch_metrics["total_proposed"] += (
                    draft_tokens.shape[0] * draft_tokens.shape[1]
                )
                batch_metrics["total_draft_time_ms"] += draft_time_ms
                batch_metrics["total_verification_time_ms"] += verify_time_ms

                # Validate generated tokens using centralized validation
                base_vocab_size = get_vocab_size(self.base_lm)
                if base_vocab_size is not None:
                    base_tokens = validate_and_clamp_tokens(
                        base_tokens, base_vocab_size, "base_output"
                    )

                draft_vocab_size = get_vocab_size(self.draft_lm)
                if draft_vocab_size is not None:
                    draft_tokens = validate_and_clamp_tokens(
                        draft_tokens, draft_vocab_size, "draft_output"
                    )

                # Debug: log batch metrics after each step (gated behind debug flag)
                if enable_debug_prints and (step == 1 or step % 16 == 0):
                    print(
                        f"[DEBUG] Step {step} metrics - "
                        f"proposed: {batch_metrics['total_proposed']}, "
                        f"accepted: {batch_metrics['total_accepted']}, "
                        f"draft_time_total: {batch_metrics['total_draft_time_ms']:.2f}ms, "
                        f"verify_time_total: {batch_metrics['total_verification_time_ms']:.2f}ms",
                        flush=True,
                    )

                # Debug: Verify KV cache alignment (first step only, small batches)
                debug_kv_verification = (
                    os.getenv("SPECDEC_DEBUG_KV_VERIFY", "0").lower()
                    in ("1", "true", "yes")
                    and step == 1
                    and batch_size <= 3
                    and kv_cache_enabled
                )

                if debug_kv_verification:
                    try:
                        # Get KV cache from speculative decoding
                        speculative_kv = self.kv_cache_manager.get_base_past_kv(
                            relative_indices_tensor
                        )

                        # Run target-only decoding for comparison
                        target_only_kv = debug_verify_kv_cache_step(
                            base_model=self.base_lm,
                            input_ids=active_input_ids,
                            attention_mask=active_attention_mask,
                            position_ids=active_position_ids,
                            num_tokens=k,
                            temperature=temperature,
                            do_sample=do_sample,
                            device=torch.device(self.device),
                        )

                        # Compare KV caches
                        matches, errors = verify_kv_cache_alignment(
                            target_only_kv=target_only_kv,
                            speculative_kv=speculative_kv,
                        )

                        if enable_debug_prints:
                            if matches:
                                print(
                                    "[KV VERIFY] ✅ KV cache alignment verified: "
                                    "target-only and speculative match",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[KV VERIFY] ❌ KV cache alignment failed: {len(errors)} errors",
                                    flush=True,
                                )
                                for error in errors[:5]:  # Show first 5 errors
                                    print(f"  - {error}", flush=True)

                            # Print checksums for debugging
                            target_checksum = compute_kv_checksum(target_only_kv)
                            spec_checksum = compute_kv_checksum(speculative_kv)
                            print(
                                f"[KV VERIFY] Target-only checksum: {target_checksum}",
                                flush=True,
                            )
                            print(
                                f"[KV VERIFY] Speculative checksum: {spec_checksum}",
                                flush=True,
                            )
                    except Exception as e:
                        if enable_debug_prints:
                            print(
                                f"[KV VERIFY] Error during verification: {e}",
                                flush=True,
                            )
                            import traceback

                            traceback.print_exc()

                # Step 3: Apply acceptance policy in batch
                accepted_lengths = []
                accepted_tokens_list = []

                # Debug: log verify outputs shape (gated behind debug flag)
                if enable_debug_prints and (step == 1 or step % 16 == 0):
                    print(
                        f"[DEBUG] Verify execution - base_tokens shape: {base_tokens.shape}, "
                        f"base_logits shape: {base_logits.shape}, "
                        f"verify_time: {verify_time_ms:.2f}ms",
                        flush=True,
                    )
                if enable_debug_prints and step == 1:
                    # Decode verify tokens (only first step)
                    if tokenizer is not None:
                        try:
                            verify_text = tokenizer.decode(base_tokens[0])
                            print(
                                f"[DEBUG] Verify tokens decoded (first prompt): {verify_text[:100]}...",
                                flush=True,
                            )
                        except Exception as e:
                            print(
                                f"[DEBUG] Failed to decode verify tokens: {e}",
                                flush=True,
                            )

                # ZERO-COPY POINTER ROLLBACK: Process acceptance with fast rewind
                # Strategy:
                # 1. Base model writes all K tokens optimistically to cache
                # 2. After acceptance, update current_seq_lens pointer (fast rewind)
                # 3. No expensive unpad/repad operations - just pointer manipulation
                kv_cache_reset_needed = False

                # Track original sequence lengths before optimistic write
                # These are the lengths BEFORE the base model wrote K tokens
                original_seq_lens = [
                    current_seq_lens[global_idx] for global_idx in active_indices
                ]

                for idx_in_active, global_idx in enumerate(active_indices):
                    # Extract tokens/logits for this prompt
                    prompt_draft_tokens = draft_tokens[
                        idx_in_active : idx_in_active + 1
                    ]
                    prompt_base_tokens = base_tokens[idx_in_active : idx_in_active + 1]
                    prompt_draft_logits = draft_logits[
                        idx_in_active : idx_in_active + 1
                    ]
                    prompt_base_logits = base_logits[idx_in_active : idx_in_active + 1]

                    # Apply policy
                    # Debug: log tokens before policy (gated behind debug flag)
                    # Minimize CPU transfers - only when debugging
                    if enable_debug_prints and step <= 2 and idx_in_active == 0:
                        # Get base predicted tokens from logits for comparison
                        # Keep operations on GPU, only transfer minimal data to CPU
                        max_debug_len = min(3, prompt_base_logits.shape[1])
                        base_pred_from_logits_gpu = torch.argmax(
                            prompt_base_logits[0, :max_debug_len, :], dim=-1
                        )
                        draft_tokens_sample_gpu = prompt_draft_tokens[
                            0, : min(3, prompt_draft_tokens.shape[1])
                        ]
                        # Only transfer small samples to CPU for logging
                        base_pred_from_logits = base_pred_from_logits_gpu.cpu().tolist()
                        draft_tokens_sample = draft_tokens_sample_gpu.cpu().tolist()
                        print(
                            f"[DEBUG] Policy input - draft_tokens[0]: {draft_tokens_sample}, "
                            f"base_argmax[0]: {base_pred_from_logits}",
                            flush=True,
                        )

                        # Calculate overlap ratio for debug
                        if (
                            len(draft_tokens_sample) > 0
                            and len(base_pred_from_logits) > 0
                        ):
                            match_count: int = sum(
                                1
                                for i in range(
                                    min(
                                        len(draft_tokens_sample),
                                        len(base_pred_from_logits),
                                    )
                                )
                                if draft_tokens_sample[i] == base_pred_from_logits[i]
                            )
                            overlap_ratio = match_count / max(
                                len(draft_tokens_sample), len(base_pred_from_logits)
                            )
                            min_len = min(
                                len(draft_tokens_sample), len(base_pred_from_logits)
                            )
                            print(
                                f"[DEBUG] Token overlap ratio: {overlap_ratio:.2f} "
                                f"({match_count}/{min_len} match)",
                                flush=True,
                            )

                    accepted_len, policy_info = self.policy.accept_tokens(
                        prompt_draft_tokens,
                        prompt_base_tokens,
                        prompt_draft_logits,
                        prompt_base_logits,
                    )
                    policy_accept_len = accepted_len

                    if policy_accept_len < prompt_draft_tokens.shape[1]:
                        kv_cache_reset_needed = True

                    # Debug: log policy result (gated behind debug flag)
                    if enable_debug_prints and (step <= 2 or step % 8 == 0):
                        proposed_count = prompt_draft_tokens.shape[1]
                        accept_rate = accepted_len / max(proposed_count, 1)
                        print(
                            f"[DEBUG] Step {step} | Proposed={proposed_count} | Accepted={accepted_len} | "
                            f"AcceptRate={accept_rate:.2%} | Policy={policy_info.get('verify_backend', 'unknown')}",
                            flush=True,
                        )

                    # Get accepted tokens - ensure we always have something
                    # CRITICAL FIX: Use base model's generated tokens, not draft tokens
                    # This ensures correctness - base model tokens are the ground truth
                    # Even if draft tokens "match" via argmax comparison, we must use base tokens
                    # to maintain correct generation chain and prevent accumulation errors
                    # OPTIMIZATION: Keep tensors on GPU as long as possible, defer CPU transfer
                    accepted_tokens_tensor = None
                    if accepted_len > 0:
                        # CRITICAL: Ensure we're using the correct slice from base tokens
                        # prompt_base_tokens shape: [1, k] where k is the number of tokens generated
                        # We need to extract exactly accepted_len tokens from the base model output
                        if accepted_len > prompt_base_tokens.shape[1]:
                            self.logger.error(
                                f"CRITICAL: accepted_len ({accepted_len}) > base_tokens.shape[1] "
                                f"({prompt_base_tokens.shape[1]}) for prompt {global_idx}"
                            )
                            accepted_len = prompt_base_tokens.shape[1]

                        # Use base model's actual generated tokens for accepted positions
                        # Extract from [batch_idx=0, :accepted_len] to get the first accepted_len tokens
                        accepted_tokens_tensor = prompt_base_tokens[
                            0, :accepted_len
                        ].clone()

                        # CRITICAL DEBUG: Verify we're not accidentally using draft tokens
                        # OPTIMIZATION: Only do this check if debug mode is enabled to avoid CPU transfers
                        if (
                            step <= 2
                            and idx_in_active == 0
                            and os.getenv("SPECDEC_DEBUG", "0").lower()
                            in ("1", "true", "yes")
                        ):
                            # Compare on GPU first to avoid unnecessary CPU transfer
                            max_cmp_len = min(
                                3, accepted_len, prompt_draft_tokens.shape[1]
                            )
                            if max_cmp_len > 0:
                                draft_sample_gpu = prompt_draft_tokens[0, :max_cmp_len]
                                base_sample_gpu = accepted_tokens_tensor[:max_cmp_len]
                                # Check on GPU
                                tokens_match = torch.all(
                                    draft_sample_gpu == base_sample_gpu
                                )
                                if tokens_match:
                                    # Only transfer to CPU if there's a potential issue
                                    draft_sample = draft_sample_gpu.cpu().tolist()
                                    base_sample = base_sample_gpu.cpu().tolist()
                                    self.logger.warning(
                                        f"WARNING: Draft and base tokens match exactly - draft: {draft_sample}, base: {base_sample}"
                                    )

                        # Validate before CPU transfer (stays on GPU)
                        base_vocab_size = get_vocab_size(self.base_lm)
                        if base_vocab_size is not None:
                            accepted_tokens_tensor = validate_and_clamp_tokens(
                                accepted_tokens_tensor,
                                base_vocab_size,
                                "accepted_tokens",
                            )
                        # OPTIMIZATION: Defer CPU transfer - keep on GPU until needed
                        # We'll transfer to CPU only when we need to check for EOS or extend list

                        # Stop before adding EOS token if encountered
                        # Get EOS token ID for early stopping
                        eos_token_id = 50256  # Default GPT-2 EOS token
                        if tokenizer is not None:
                            eos_token_id = tokenizer.eos_token_id
                        elif (
                            hasattr(self.base_lm, "_tokenizer")
                            and self.base_lm._tokenizer is not None
                        ):
                            base_tokenizer = self.base_lm._tokenizer
                            if base_tokenizer is not None and hasattr(
                                base_tokenizer, "eos_token_id"
                            ):
                                eos_token_id = base_tokenizer.eos_token_id

                        # OPTIMIZATION: Check for EOS on GPU first (faster than CPU)
                        # Only transfer to CPU if EOS found or needed for list operations
                        if accepted_tokens_tensor is not None:
                            # Check for EOS on GPU (no CPU sync)
                            eos_mask = accepted_tokens_tensor == eos_token_id
                            if eos_mask.any():
                                # EOS found - find first occurrence on GPU
                                eos_positions = torch.nonzero(eos_mask, as_tuple=False)
                                if len(eos_positions) > 0:
                                    # Use GPU-only operation to get index, only .item() when necessary
                                    eos_idx: int = int(eos_positions[0].item())
                                    # Truncate at EOS (still on GPU)
                                    accepted_tokens_tensor = accepted_tokens_tensor[
                                        :eos_idx
                                    ]
                                    batch_active[global_idx] = False  # Stop generation

                            # EQSPEC BONUS TOKEN: Sample exactly one token from target model distribution
                            # at the mismatch position (after accepted draft tokens)
                            # DRAFT CACHE SYNCHRONIZATION: Append bonus token to draft cache
                            bonus_token_tensor = None
                            k = prompt_draft_tokens.shape[
                                1
                            ]  # Number of draft tokens generated

                            if accepted_len < k:
                                # We have logits at the mismatch position from verification
                                # Use prompt_base_logits[0, accepted_len, :] for bonus token
                                bonus_logits = prompt_base_logits[
                                    0, accepted_len, :
                                ]  # [vocab_size]

                                # Get generation parameters from kwargs or config
                                top_p = kwargs.get(
                                    "top_p", self.config.get("top_p", None)
                                )
                                top_k = kwargs.get(
                                    "top_k", self.config.get("top_k", None)
                                )

                                # Sample bonus token from target model distribution
                                bonus_token_tensor = sample_bonus_token_from_logits(
                                    logits=bonus_logits,
                                    temperature=temperature,
                                    do_sample=do_sample,
                                    top_p=top_p,
                                    top_k=top_k,
                                    vocab_size=base_vocab_size,
                                )  # [1]

                                if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                ):
                                    self.logger.debug(
                                        f"EQSPEC: Sampled bonus token {bonus_token_tensor.item()} "
                                        f"from existing logits at position {accepted_len}"
                                    )
                            elif accepted_len == k:
                                # All draft tokens accepted - need forward pass for bonus token
                                # Construct input with accepted tokens
                                current_seq = current_input_ids[
                                    global_idx
                                ]  # Unpadded sequence
                                bonus_input = torch.cat(
                                    [current_seq, accepted_tokens_tensor], dim=0
                                ).unsqueeze(
                                    0
                                )  # [1, seq_len + accepted_len]

                                # Forward pass on target model to get bonus token logits
                                with torch.no_grad():
                                    # Get past_key_values if KV cache is enabled
                                    bonus_past_kv = None
                                    if kv_cache_enabled:
                                        # Get KV cache for this specific sequence
                                        # Note: We need to get KV cache up to current_seq + accepted_tokens
                                        # For now, we'll do a fresh forward pass
                                        # TODO: Optimize to reuse KV cache if possible
                                        pass

                                    # Forward pass to get logits at the last position
                                    if hasattr(self.base_lm, "_model"):
                                        model_outputs = self.base_lm._model(
                                            input_ids=bonus_input,
                                            past_key_values=bonus_past_kv,
                                            use_cache=False,
                                        )
                                        bonus_logits = model_outputs.logits[
                                            0, -1, :
                                        ]  # [vocab_size]
                                    else:
                                        # Fallback: use generate_tokens with max_new_tokens=1
                                        # but we only need the logits, not the token
                                        # This is less efficient but works as fallback
                                        _, bonus_logits_full = (
                                            self.base_lm.generate_tokens(
                                                bonus_input,
                                                max_new_tokens=1,
                                                temperature=temperature,
                                                do_sample=False,  # We'll sample ourselves
                                            )
                                        )
                                        # bonus_logits_full is [1, 1, vocab_size], extract [vocab_size]
                                        bonus_logits = bonus_logits_full[0, 0, :]

                                # Get generation parameters
                                top_p = kwargs.get(
                                    "top_p", self.config.get("top_p", None)
                                )
                                top_k = kwargs.get(
                                    "top_k", self.config.get("top_k", None)
                                )

                                # Sample bonus token
                                bonus_token_tensor = sample_bonus_token_from_logits(
                                    logits=bonus_logits,
                                    temperature=temperature,
                                    do_sample=do_sample,
                                    top_p=top_p,
                                    top_k=top_k,
                                    vocab_size=base_vocab_size,
                                )  # [1]

                                if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                ):
                                    self.logger.debug(
                                        f"EQSPEC: Sampled bonus token {bonus_token_tensor.item()} "
                                        f"from forward pass (all {k} draft tokens accepted)"
                                    )

                            # Append bonus token to accepted tokens if we have one
                            if bonus_token_tensor is not None:
                                # Validate bonus token
                                if base_vocab_size is not None:
                                    bonus_token_tensor = validate_and_clamp_tokens(
                                        bonus_token_tensor,
                                        base_vocab_size,
                                        "bonus_token",
                                    )

                                # Check for EOS
                                eos_token_id = 50256  # Default GPT-2 EOS token
                                if tokenizer is not None:
                                    eos_token_id = tokenizer.eos_token_id
                                elif (
                                    hasattr(self.base_lm, "_tokenizer")
                                    and self.base_lm._tokenizer is not None
                                ):
                                    base_tokenizer = self.base_lm._tokenizer
                                    if base_tokenizer is not None and hasattr(
                                        base_tokenizer, "eos_token_id"
                                    ):
                                        eos_token_id = base_tokenizer.eos_token_id

                                if bonus_token_tensor.item() == eos_token_id:
                                    batch_active[global_idx] = False

                                # Append bonus token to accepted tokens tensor
                                accepted_tokens_tensor = torch.cat(
                                    [accepted_tokens_tensor, bonus_token_tensor], dim=0
                                )  # [accepted_len + 1]

                                if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                ):
                                    self.logger.debug(
                                        f"EQSPEC: Appended bonus token to accepted tokens. "
                                        f"Total accepted tokens: {accepted_tokens_tensor.shape[0]}"
                                    )

                                # DRAFT CACHE SYNCHRONIZATION: Append bonus token to draft cache
                                # This ensures the draft model has the correct context for next step
                                if kv_cache_enabled and bonus_token_tensor is not None:
                                    # Get the draft cache's current length for this sequence
                                    draft_current_len = (
                                        self.kv_cache_manager.draft_current_seq_lens[
                                            idx_in_active
                                        ]
                                        if idx_in_active
                                        < len(
                                            self.kv_cache_manager.draft_current_seq_lens
                                        )
                                        else current_seq_lens[global_idx]
                                    )

                                    # We need to append the bonus token's KV to the draft cache
                                    # For now, we'll update the draft cache pointer to include the bonus token
                                    # The actual KV will be computed in the next draft step
                                    # But we need to ensure the draft cache length matches the base cache
                                    # This is a simplified approach - in a full implementation, we'd
                                    # compute the bonus token's KV and append it using kv_append_inplace

                                    # Update draft cache pointer to match base cache (including bonus)
                                    # The draft model will generate from this position next step
                                    if idx_in_active < len(
                                        self.kv_cache_manager.draft_current_seq_lens
                                    ):
                                        # Draft cache should be at: original_len + accepted_len + 1 (bonus)
                                        new_draft_seq_len = (
                                            original_seq_lens[idx_in_active]
                                            + accepted_len
                                            + 1
                                        )
                                        self.kv_cache_manager.draft_current_seq_lens[
                                            idx_in_active
                                        ] = new_draft_seq_len

                                        if enable_debug_prints and step <= 2:
                                            print(
                                                f"[DRAFT_SYNC] Prompt {global_idx}: "
                                                f"Appended bonus token to draft cache, "
                                                f"new_draft_len={new_draft_seq_len}",
                                                flush=True,
                                            )

                            # OPTIMIZATION: Single CPU transfer for all operations
                            # Convert to list ONCE at the end
                            accepted_tokens = accepted_tokens_tensor.cpu().tolist()

                            # CRITICAL: Check for duplication BEFORE adding to batch_generated_tokens
                            # This prevents repetitive text patterns like "TheTheTheTheThe"
                            # OPTIMIZATION: Simplify duplication detection - only check for direct overlap
                            # This reduces CPU overhead while still catching most duplication issues
                            if (
                                len(batch_generated_tokens[global_idx]) > 0
                                and len(accepted_tokens) > 0
                            ):
                                generated_so_far = batch_generated_tokens[global_idx]

                                # Only check for phrase repetition (covers single token repetition too)
                                # Check if last N tokens of generated match first N tokens of accepted
                                for check_len in range(
                                    min(5, len(generated_so_far), len(accepted_tokens)),
                                    0,
                                    -1,
                                ):
                                    generated_tail = generated_so_far[-check_len:]
                                    accepted_head = accepted_tokens[:check_len]
                                    if generated_tail == accepted_head:
                                        # Found duplication - skip the duplicate tokens
                                        if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                            "1",
                                            "true",
                                            "yes",
                                        ):
                                            self.logger.warning(
                                                f"CRITICAL: Detected {check_len}-token overlap "
                                                f"for prompt {global_idx} at step {step}. Skipping duplicates."
                                            )
                                        # Skip the duplicate tokens
                                        accepted_tokens = accepted_tokens[check_len:]
                                        break

                            tokens_to_add = accepted_tokens
                        else:
                            tokens_to_add = []

                        # Only add non-empty tokens
                        if tokens_to_add:
                            batch_generated_tokens[global_idx].extend(tokens_to_add)
                        accepted_tokens_list.append(tokens_to_add)
                        # Update accepted_tokens for use in sequence update below
                        accepted_tokens = tokens_to_add
                        accepted_len = len(tokens_to_add)
                    else:
                        # Rejected all draft tokens - sample from first base token distribution
                        # EQSPEC: When no draft tokens are accepted, we still need to advance
                        # by sampling from the target model distribution at position 0
                        base_vocab_size = get_vocab_size(self.base_lm)

                        # Get logits at position 0 from verification
                        first_base_logits = prompt_base_logits[0, 0, :]  # [vocab_size]

                        # Get generation parameters
                        top_p = kwargs.get("top_p", self.config.get("top_p", None))
                        top_k = kwargs.get("top_k", self.config.get("top_k", None))

                        # Sample token from target model distribution
                        first_base_tensor = sample_bonus_token_from_logits(
                            logits=first_base_logits,
                            temperature=temperature,
                            do_sample=do_sample,
                            top_p=top_p,
                            top_k=top_k,
                            vocab_size=base_vocab_size,
                        )  # [1]

                        # Validate with centralized validation (stays on GPU)
                        if base_vocab_size is not None:
                            first_base_tensor = validate_and_clamp_tokens(
                                first_base_tensor, base_vocab_size, "fallback_base"
                            )

                        # OPTIMIZATION: Check for EOS on GPU before CPU transfer
                        eos_token_id = 50256  # Default GPT-2 EOS token
                        if tokenizer is not None:
                            eos_token_id = tokenizer.eos_token_id
                        elif (
                            hasattr(self.base_lm, "_tokenizer")
                            and self.base_lm._tokenizer is not None
                        ):
                            base_tokenizer = self.base_lm._tokenizer
                            if base_tokenizer is not None and hasattr(
                                base_tokenizer, "eos_token_id"
                            ):
                                eos_token_id = base_tokenizer.eos_token_id

                        # Check for EOS on GPU (no CPU sync)
                        if (first_base_tensor == eos_token_id).any():
                            batch_active[global_idx] = False  # Stop generation

                        # OPTIMIZATION: Single CPU transfer
                        accepted_tokens = first_base_tensor.cpu().tolist()

                        # Don't add EOS token if it's the fallback token
                        if accepted_tokens and accepted_tokens[0] == eos_token_id:
                            accepted_tokens = []
                            batch_active[global_idx] = False

                        # CRITICAL: Check for duplication in fallback case too
                        if accepted_tokens:
                            # Check for duplication before adding
                            generated_so_far = batch_generated_tokens[global_idx]
                            if len(generated_so_far) > 0 and len(accepted_tokens) > 0:
                                if generated_so_far[-1] == accepted_tokens[0]:
                                    self.logger.warning(
                                        f"CRITICAL: Detected token duplication in fallback for prompt {global_idx} at step {step}. "
                                        f"Skipping duplicate token {accepted_tokens[0]}"
                                    )
                                    accepted_tokens = []

                            if accepted_tokens:
                                batch_generated_tokens[global_idx].extend(
                                    accepted_tokens
                                )
                            accepted_tokens_list.append(accepted_tokens)
                            accepted_len = len(accepted_tokens)
                        else:
                            accepted_len = 0
                        if enable_debug_prints and (step <= 3 or idx_in_active == 0):
                            print(
                                f"[DEBUG] No draft tokens accepted - accepting first base token: {accepted_tokens}",
                                flush=True,
                            )

                    # Debug: log acceptance per prompt (gated behind debug flag)
                    if enable_debug_prints and (step <= 2 or step % 8 == 0):
                        print(
                            f"[DEBUG] Acceptance - prompt {global_idx}, "
                            f"proposed={prompt_draft_tokens.shape[1]}, "
                            f"accepted={accepted_len}, "
                            f"tokens: {accepted_tokens[:min(5, len(accepted_tokens))]}",
                            flush=True,
                        )

                    accepted_lengths.append(accepted_len)
                    batch_metrics["total_accepted"] += accepted_len
                    batch_metrics[
                        "total_generated_tokens"
                    ] += accepted_len  # Track total generated

                    # Track per-prompt metrics for accurate reporting
                    per_prompt_proposed_counts[global_idx] += prompt_draft_tokens.shape[
                        1
                    ]
                    per_prompt_accepted_counts[global_idx] += accepted_len

                    # ZERO-COPY POINTER ROLLBACK: Fast Rewind Logic
                    # The base model has already written all K tokens optimistically to the cache.
                    # Now we update the pointer to reflect only accepted tokens.
                    # This avoids expensive unpad/repad operations.
                    if kv_cache_enabled:
                        # Get the original sequence length (before optimistic write)
                        original_len = original_seq_lens[idx_in_active]

                        # FAST REWIND: Update pointer to reflect only accepted tokens
                        # If num_accepted < K, we simply decrement the pointer
                        # The rejected tokens remain in the buffer but are "hidden" by the pointer
                        new_base_seq_len = original_len + accepted_len

                        # Update cache manager's pointer (this is the fast rewind)
                        if idx_in_active < len(
                            self.kv_cache_manager.base_current_seq_lens
                        ):
                            self.kv_cache_manager.base_current_seq_lens[
                                idx_in_active
                            ] = new_base_seq_len

                        # Update our tracking pointer
                        current_seq_lens[global_idx] = new_base_seq_len

                        if enable_debug_prints and (step <= 2 or step % 8 == 0):
                            k_generated = prompt_draft_tokens.shape[1]
                            print(
                                f"[FAST_REWIND] Prompt {global_idx}: "
                                f"original_len={original_len}, "
                                f"k_generated={k_generated}, "
                                f"accepted={accepted_len}, "
                                f"new_len={new_base_seq_len}, "
                                f"rejected={k_generated - accepted_len} (hidden by pointer)",
                                flush=True,
                            )

                    # Update current input for next iteration
                    # Concatenate accepted tokens to current input (no padding, keep as 1D)
                    # Use accepted_tokens directly (already extracted above)
                    if len(accepted_tokens) > 0:
                        # CRITICAL: Validate accepted tokens one more time before creating tensor
                        # This ensures no invalid indices are introduced during list operations
                        if hasattr(self.base_lm, "_model") and hasattr(
                            self.base_lm._model, "config"
                        ):
                            vocab_size = getattr(
                                self.base_lm._model.config, "vocab_size", None
                            )
                            if vocab_size is not None:
                                # Ensure all tokens are valid integers in range
                                accepted_tokens = [
                                    max(0, min(int(tok), vocab_size - 1))
                                    for tok in accepted_tokens
                                ]

                        accepted_tokens_tensor = torch.tensor(
                            accepted_tokens, device=self.device, dtype=torch.long
                        )  # Shape: [accepted_len]

                        # Validate tokens before concatenation with centralized validation
                        base_vocab_size = get_vocab_size(self.base_lm)
                        if base_vocab_size is not None:
                            accepted_tokens_tensor = validate_and_clamp_tokens(
                                accepted_tokens_tensor,
                                base_vocab_size,
                                f"accepted_tensor_{global_idx}",
                            )

                        current_seq = current_input_ids[global_idx]  # Shape: [seq_len]

                        # CRITICAL: Validate current sequence with DRAFT vocab size
                        # Since this will be used for draft model in next iteration
                        draft_vocab_size = get_vocab_size(self.draft_lm)
                        if draft_vocab_size is not None:
                            current_seq = validate_and_clamp_tokens(
                                current_seq,
                                draft_vocab_size,
                                f"current_seq_{global_idx}",
                            )

                        # Debug: log before/after append (gated behind debug flag)
                        if enable_debug_prints and step == 1:
                            print(
                                f"[DEBUG] Sequence update - prompt {global_idx}: "
                                f"before_len={current_seq.shape[0]}, "
                                f"accepted_len={len(accepted_tokens)}, "
                                f"after_len={current_seq.shape[0] + len(accepted_tokens)}",
                                flush=True,
                            )

                        # CRITICAL: Check for token duplication before concatenation on GPU
                        # This prevents repetitive text generation bugs
                        # OPTIMIZATION: Do all comparisons on GPU, only transfer to CPU if issue detected
                        if (
                            current_seq.shape[0] > 0
                            and accepted_tokens_tensor.shape[0] > 0
                        ):
                            # Check on GPU first - avoid unnecessary CPU transfers
                            check_len = min(
                                5,  # Check up to 5 tokens for overlap
                                current_seq.shape[0],
                                accepted_tokens_tensor.shape[0],
                            )
                            if check_len > 0:
                                # Compare tensors on GPU
                                current_tail = current_seq[-check_len:]
                                accepted_head = accepted_tokens_tensor[:check_len]
                                tokens_match = torch.all(current_tail == accepted_head)

                                if tokens_match:
                                    # Only log and transfer to CPU if duplication detected
                                    if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                        "1",
                                        "true",
                                        "yes",
                                    ):
                                        current_tail_list = current_tail.cpu().tolist()
                                        accepted_head_list = (
                                            accepted_head.cpu().tolist()
                                        )
                                        self.logger.warning(
                                            f"CRITICAL: Detected {check_len}-token duplication for prompt {global_idx} at step {step}! "
                                            f"Tail: {current_tail_list}, Head: {accepted_head_list}. Skipping duplicates."
                                        )

                                    # Skip the duplicate tokens - only add new ones
                                    if accepted_tokens_tensor.shape[0] > check_len:
                                        accepted_tokens_tensor = accepted_tokens_tensor[
                                            check_len:
                                        ]
                                        # Update accepted_tokens list to match
                                        accepted_tokens = (
                                            accepted_tokens_tensor.cpu().tolist()
                                        )
                                        tokens_to_add = accepted_tokens
                                        # Update the generated tokens list to remove duplicates
                                        if (
                                            len(batch_generated_tokens[global_idx])
                                            >= check_len
                                        ):
                                            batch_generated_tokens[global_idx] = (
                                                batch_generated_tokens[global_idx][
                                                    :-check_len
                                                ]
                                            )
                                    else:
                                        # All tokens are duplicates, skip this update
                                        if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                            "1",
                                            "true",
                                            "yes",
                                        ):
                                            self.logger.warning(
                                                f"All accepted tokens are duplicates, skipping sequence update for prompt {global_idx}"
                                            )
                                        accepted_tokens_tensor = torch.tensor(
                                            [], device=self.device, dtype=torch.long
                                        )
                                        accepted_tokens = []
                                        tokens_to_add = []

                        # Concatenate along sequence dimension (both are 1D)
                        # Only concatenate if we have tokens to add
                        if accepted_tokens_tensor.shape[0] > 0:
                            updated_seq = torch.cat(
                                [current_seq, accepted_tokens_tensor], dim=0
                            )
                        else:
                            # No new tokens to add, keep current sequence
                            updated_seq = current_seq

                        # CRITICAL: Validate after concatenation with DRAFT vocab size
                        # Since this will be used for draft model in next iteration
                        draft_vocab_size = get_vocab_size(self.draft_lm)
                        if draft_vocab_size is not None:
                            updated_seq = validate_and_clamp_tokens(
                                updated_seq,
                                draft_vocab_size,
                                f"updated_seq_{global_idx}",
                            )

                        # CRITICAL: Clone before storing to ensure independence
                        # This prevents corruption if tensor is still being used elsewhere
                        # Same pattern as KV cache updates: detach().contiguous() for safety
                        updated_seq = updated_seq.detach().clone().contiguous()

                        # CRITICAL FIX: Ensure GPU operations complete before updating sequence
                        # This prevents race conditions where sequence update happens before
                        # GPU operations (concatenation, cloning) complete, which can cause
                        # token duplication and notebook unresponsiveness.
                        # Only sync if periodic sync enabled - detach().clone() is usually sufficient.
                        if (
                            enable_periodic_sync
                            and self.device == "cuda"
                            and torch.cuda.is_available()
                        ):
                            # Quick sync to ensure concatenation and cloning are complete
                            torch.cuda.synchronize()

                        # Update sequence (keep as 1D list item, no padding)
                        current_input_ids[global_idx] = updated_seq
                    else:
                        # No tokens accepted - log warning but continue with next base token
                        if enable_debug_prints:
                            print(
                                f"[WARNING] Step {step}, prompt {global_idx}: No tokens to append!",
                                flush=True,
                            )

                    # Check if this prompt is done (additional check for max_tokens)
                    # EOS handling is already done above when adding tokens
                    if len(batch_generated_tokens[global_idx]) >= max_tokens:
                        batch_active[global_idx] = False

                # ZERO-COPY POINTER ROLLBACK: No realignment needed!
                # With pointer-based rollback, we don't need expensive realign operations.
                # The current_seq_lens pointers are already updated above (fast rewind).
                # The cache manager's get_base_past_kv() and get_draft_past_kv() methods
                # automatically return sliced views based on current_seq_lens.
                #
                # Old approach (expensive):
                #   - realign_kv_cache() would clone, stack, pad, and rearrange tensors
                #   - O(N) memory operations per rejection
                #
                # New approach (zero-copy):
                #   - Just update current_seq_lens pointer (O(1))
                #   - get_base_past_kv() returns a view sliced to current_seq_len
                #   - No memory operations, no unpad/repad overhead
                #
                # Note: We still update sequence_lengths dict for compatibility,
                # but this is just a dict update (O(1)), not tensor operations.
                if kv_cache_enabled:
                    # Update sequence_lengths dict for compatibility (lightweight dict update)
                    for idx_in_active, global_idx in enumerate(active_indices):
                        if batch_active[global_idx]:  # Only update active sequences
                            global_id = global_sequence_ids[global_idx]
                            # Use current_seq_lens as source of truth
                            self.kv_cache_manager.base_sequence_lengths[global_id] = (
                                current_seq_lens[global_idx]
                            )
                            # Draft cache length matches base (including bonus token if added)
                            if idx_in_active < len(
                                self.kv_cache_manager.draft_current_seq_lens
                            ):
                                self.kv_cache_manager.draft_sequence_lengths[
                                    global_id
                                ] = self.kv_cache_manager.draft_current_seq_lens[
                                    idx_in_active
                                ]
                            else:
                                self.kv_cache_manager.draft_sequence_lengths[
                                    global_id
                                ] = current_seq_lens[global_idx]

                batch_metrics["total_steps"] += 1

                # Log sequence pool statistics periodically (gated behind debug flag)
                if (
                    enable_debug_prints
                    and sequence_pool is not None
                    and (step == 1 or step % 10 == 0)
                ):
                    pool_stats = sequence_pool.get_statistics()
                    print(
                        f"[SEQUENCE_POOL] Step {step}: "
                        f"groups={pool_stats['total_groups_formed']}, "
                        f"same_length={pool_stats['same_length_percentage']:.1f}%, "
                        f"mixed_length={pool_stats['mixed_length_percentage']:.1f}%, "
                        f"avg_group_size={pool_stats['avg_group_size']:.1f}",
                        flush=True,
                    )

                if kv_cache_reset_needed and kv_cache_enabled:
                    if enable_debug_prints:
                        print(
                            "[BATCH] Disabling KV cache reuse after partial acceptance to maintain consistency",
                            flush=True,
                        )
                    self.kv_cache_manager.reset()
                    kv_cache_enabled = False
                    if hasattr(self.base_lm, "clear_kv_cache"):
                        self.base_lm.clear_kv_cache()
                    if hasattr(self.draft_lm, "clear_kv_cache"):
                        self.draft_lm.clear_kv_cache()

                # OPTIMIZATION: Only synchronize periodically, not every iteration
                # Excessive synchronization causes high CPU usage (100%) and slow performance
                # Use CUDA events for async execution instead of barrier synchronization
                # Only synchronize when necessary to reduce CPU-GPU overhead
                if (
                    enable_periodic_sync
                    and self.device == "cuda"
                    and torch.cuda.is_available()
                ):
                    # Only synchronize every 10 steps or on last step to reduce CPU overhead
                    # Event-based sync in scheduler handles async correctness
                    if step % 10 == 0 or not any(batch_active):
                        torch.cuda.synchronize()

                # OPTIMIZATION: Clear CUDA cache periodically to prevent memory leak
                # Clear every 20 steps to prevent gradual GPU memory accumulation
                # WARNING: This can cause stalls - only enable if memory is an issue
                if (
                    enable_periodic_cache_clear
                    and self.device == "cuda"
                    and torch.cuda.is_available()
                ):
                    if step % 20 == 0:
                        torch.cuda.empty_cache()
                        # Also clear KV caches periodically if they exist
                        if not kv_cache_reset_needed:  # Don't double-clear
                            if hasattr(self.base_lm, "clear_kv_cache"):
                                self.base_lm.clear_kv_cache()
                            if hasattr(self.draft_lm, "clear_kv_cache"):
                                self.draft_lm.clear_kv_cache()

                # Log batch progress with accurate timing (only if SPECDEC_DEBUG enabled)
                if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                    if step % 8 == 0 or step == 1:
                        avg_draft_time = (
                            batch_metrics["total_draft_time_ms"]
                            / batch_metrics["total_steps"]
                            if batch_metrics["total_steps"] > 0
                            else 0.0
                        )
                        avg_verify_time = (
                            batch_metrics["total_verification_time_ms"]
                            / batch_metrics["total_steps"]
                            if batch_metrics["total_steps"] > 0
                            else 0.0
                        )
                        print(
                            f"[BATCH] Step {step}/{max_tokens} | "
                            f"Active: {active_count}/{batch_size} | "
                            f"K={k} | "
                            f"Draft: {draft_time_ms:.1f}ms (avg={avg_draft_time:.1f}ms) | "
                            f"Verify: {verify_time_ms:.1f}ms (avg={avg_verify_time:.1f}ms) | "
                            f"Accepted: {sum(accepted_lengths)}/{len(accepted_lengths)*k}",
                            flush=True,
                        )

            total_time_ms = (time.time() - generation_start) * 1000
            batch_metrics["total_generation_time_ms"] = total_time_ms
            batch_metrics["total_time_ms"] = (
                total_time_ms  # Add for throughput calculation
            )

            # Calculate batch-level throughput
            batch_metrics["tokens_per_sec"] = (
                batch_metrics["total_generated_tokens"] / (total_time_ms / 1000.0)
                if total_time_ms > 0
                else 0.0
            )

            # Log batch metrics
            self.logger.info(
                f"[METRICS] Batch total - Tokens={batch_metrics['total_generated_tokens']} "
                f"Time={total_time_ms:.2f}ms "
                f"→ {batch_metrics['tokens_per_sec']:.2f} tok/s, "
                f"Proposed={batch_metrics['total_proposed']} "
                f"Accepted={batch_metrics['total_accepted']} "
                f"AcceptRate={batch_metrics['total_accepted']/max(batch_metrics['total_proposed'], 1):.2%}"
            )
            print(
                f"[METRICS] Batch total - Tokens={batch_metrics['total_generated_tokens']} "
                f"Time={total_time_ms:.2f}ms "
                f"→ {batch_metrics['tokens_per_sec']:.2f} tok/s",
                flush=True,
            )

            # OPTIMIZATION: Cleanup CUDA resources efficiently without excessive synchronization
            # Excessive synchronization can cause notebook hanging, so we minimize it
            # Clear memory and KV caches, but avoid redundant synchronization
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Step 1: Wait for streams to complete (event-based, not barrier sync)
                    # Streams should already be synchronized from event-based approach
                    # Only do explicit sync if streams still exist
                    if draft_stream is not None:
                        draft_stream.synchronize()
                    if verify_stream is not None:
                        verify_stream.synchronize()

                    # Step 2: Clear KV caches to free model memory (no sync needed)
                    if hasattr(self.base_lm, "clear_kv_cache"):
                        self.base_lm.clear_kv_cache()
                    if hasattr(self.draft_lm, "clear_kv_cache"):
                        self.draft_lm.clear_kv_cache()
                    if hasattr(self, "kv_cache_manager"):
                        self.kv_cache_manager.reset()

                    # Step 3: Force Python garbage collection to free CUDA objects
                    import gc

                    gc.collect()

                    # Step 4: Clear CUDA cache once after GC (no sync needed)
                    torch.cuda.empty_cache()

                    # Step 5: Single final synchronization to ensure clean state
                    # Only one sync instead of multiple to prevent notebook hanging
                    torch.cuda.synchronize()

                    if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                        self.logger.debug(
                            "[CLEANUP] CUDA streams synchronized, cache cleared, KV caches reset"
                        )
                except Exception as e:
                    # Don't fail on cleanup errors, but log them
                    self.logger.warning(
                        f"[CLEANUP] Error cleaning up CUDA resources: {e}",
                        exc_info=True,
                    )

            # Log final sequence pool statistics if enabled (gated behind debug flag)
            if enable_debug_prints and sequence_pool is not None:
                pool_stats = sequence_pool.get_statistics()
                print(
                    f"[SEQUENCE_POOL] Final stats: "
                    f"total_groups={pool_stats['total_groups_formed']}, "
                    f"same_length_tokens={pool_stats['same_length_tokens']}, "
                    f"mixed_length_tokens={pool_stats['mixed_length_tokens']}, "
                    f"same_length_pct={pool_stats['same_length_percentage']:.1f}%, "
                    f"avg_group_size={pool_stats['avg_group_size']:.1f}",
                    flush=True,
                )

            # Convert batched results to per-prompt dictionaries
            results = []
            for i, (prompt, generated_tokens) in enumerate(
                zip(prompts, batch_generated_tokens)
            ):
                # Decode generated tokens
                # Pass list directly to decode - it handles conversion and skips special tokens
                if generated_tokens:
                    generated_text = self.base_lm.decode(generated_tokens)
                else:
                    generated_text = ""

                # Calculate per-prompt metrics
                prompt_acceptance_rate = batch_metrics["total_accepted"] / max(
                    batch_metrics["total_proposed"], 1
                )

                # Throughput: tokens generated for this prompt / total time
                prompt_throughput = (
                    len(generated_tokens) / (total_time_ms / 1000.0)
                    if total_time_ms > 0
                    else 0.0
                )

                # Latency: average time per token for this prompt
                prompt_latency_ms = (
                    total_time_ms / len(generated_tokens)
                    if len(generated_tokens) > 0
                    else 0.0
                )

                # Calculate averages for reporting
                avg_draft_time = (
                    batch_metrics["total_draft_time_ms"] / batch_metrics["total_steps"]
                    if batch_metrics["total_steps"] > 0
                    else 0.0
                )
                avg_verify_time = (
                    batch_metrics["total_verification_time_ms"]
                    / batch_metrics["total_steps"]
                    if batch_metrics["total_steps"] > 0
                    else 0.0
                )

                # Get kernel backend info for KV append status
                # NOTE: KV append is intentionally disabled for batch processing
                # due to complexity of per-prompt sequence state tracking
                try:
                    from kernels import get_kernel_info

                    kernel_info = get_kernel_info()
                    kv_append_backend = kernel_info.get("kv_append_backend", "unknown")
                except ImportError:
                    kv_append_backend = "unavailable"

                results.append(
                    {
                        "prompt": prompt,
                        "text": generated_text,  # Key expected by script
                        "generated_text": generated_text,
                        "generated_tokens": generated_tokens,
                        "num_generated": len(generated_tokens),
                        "batch_index": i,
                        "batch_size": batch_size,
                        "latency_ms": prompt_latency_ms,  # Key expected by script
                        "total_time_ms": total_time_ms,
                        "tokens_per_sec": prompt_throughput,  # Key expected by script
                        "throughput_tokens_per_sec": prompt_throughput,
                        "acceptance_rate": prompt_acceptance_rate,
                        "proposed": per_prompt_proposed_counts[
                            i
                        ],  # Actual proposed for this prompt
                        "accepted": per_prompt_accepted_counts[
                            i
                        ],  # Actual accepted for this prompt
                        "draft_avg_ms": avg_draft_time,
                        "verify_avg_ms": avg_verify_time,
                        "batch_metrics": batch_metrics,
                        # KV append status (disabled for batch mode by design)
                        "kv_append_enabled": False,  # Batch mode doesn't use KV append
                        "kv_append_backend": kv_append_backend,
                        "kv_appended_tokens": 0,  # No KV append in batch mode
                        "kv_append_time_ms": 0.0,
                    }
                )

        # Track memory after batch
        mem_after = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )
        mem_used_mb = (mem_after - mem_before) / (1024 * 1024)

        if mem_after > 0:
            print(
                f"[BATCH] GPU memory after: {mem_after / (1024**2):.2f} MB | "
                f"Delta: {mem_used_mb:.2f} MB",
                flush=True,
            )

        self.logger.info(
            f"Batch generation complete: {batch_size} prompts, "
            f"GPU memory used: {mem_used_mb:.2f} MB"
        )
        print(
            f"[BATCH] Batch generation complete: {batch_size} prompts processed",
            flush=True,
        )

        return results
