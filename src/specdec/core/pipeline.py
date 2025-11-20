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
from .batch_handlers import (
    AcceptanceHandler,
    DraftGenerationHandler,
    KVCacheHandler,
    RollbackHandler,
    VerificationHandler,
)
from .batch_loop import BatchGenerationLoop
from .batch_utilities import BatchMetricsCollector, BatchSequenceManager
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

    def _create_batch_handlers(self):
        """Create handler instances for batch processing."""
        from .batch_handlers import (
            AcceptanceHandler,
            DraftGenerationHandler,
            KVCacheHandler,
            RollbackHandler,
            VerificationHandler,
        )

        draft_handler = DraftGenerationHandler(
            draft_lm=self.draft_lm,
            device=self.device,
            kv_cache_manager=self.kv_cache_manager,
            logger=self.logger,
        )

        verify_handler = VerificationHandler(
            base_lm=self.base_lm,
            device=self.device,
            kv_cache_manager=self.kv_cache_manager,
            logger=self.logger,
        )

        tokenizer = (
            self.base_lm._tokenizer if hasattr(self.base_lm, "_tokenizer") else None
        )

        accept_handler = AcceptanceHandler(
            policy=self.policy,
            tokenizer=tokenizer,
            base_lm=self.base_lm,
            logger=self.logger,
        )

        rollback_handler = RollbackHandler(
            kv_cache_manager=self.kv_cache_manager,
            logger=self.logger,
        )

        kv_handler = KVCacheHandler(
            kv_cache_manager=self.kv_cache_manager,
            logger=self.logger,
        )

        return {
            "draft": draft_handler,
            "verify": verify_handler,
            "accept": accept_handler,
            "rollback": rollback_handler,
            "kv": kv_handler,
        }

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

                # CRITICAL FIX: Append draft tokens to input_ids for parallel verification
                # The draft tokens will be processed in parallel during prefill, and we only need
                # to generate 1 additional token (the bonus token) for verification
                # Handle the case where draft_tokens might be empty (k=0 or generation failure)
                if (
                    draft_tokens is not None
                    and draft_tokens.numel() > 0
                    and draft_tokens.shape[1] > 0
                ):
                    verify_input_ids = torch.cat([current_input, draft_tokens], dim=1)
                else:
                    # Fallback: if no draft tokens, use original input (K=0 case)
                    verify_input_ids = current_input
                    if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                        self.logger.debug(
                            f"Step {step}: No draft tokens available, using original input for verification"
                        )

                # Single validation point before model call
                base_vocab_size = get_vocab_size(self.base_lm)
                if base_vocab_size is not None:
                    verify_input_ids = validate_and_clamp_tokens(
                        verify_input_ids, base_vocab_size, "base_input"
                    )

                # Removed unnecessary sync - CUDA events handle synchronization

                # Use scheduler or direct call
                if self.scheduler:
                    base_tokens, base_logits, verify_info = (
                        self.scheduler.schedule_verification(
                            self.base_lm,
                            draft_tokens,  # Pass for shape info
                            verify_input_ids,  # Input with draft tokens appended
                            temperature=temperature,
                            do_sample=do_sample,
                            **kwargs,
                        )
                    )
                    verify_time_ms = verify_info.get("verification_time_ms", 0.0)
                else:
                    verify_start = time.time()
                    base_tokens, base_logits = self.base_lm.generate_tokens(
                        verify_input_ids,  # Input with draft tokens appended
                        max_new_tokens=1,  # Only generate 1 token (bonus token) - draft tokens processed in prefill
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

        Refactored version using handler classes for cleaner, maintainable code.

        Args:
            prompts: List of input prompt strings
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            List of result dictionaries, one per prompt
        """
        if not prompts:
            return []

        self.kv_cache_manager.reset()
        batch_size = len(prompts)
        self.logger.info(f"Starting batched generation for {batch_size} prompts")

        max_tokens = max_tokens or self.config["max_new_tokens"]
        temperature = temperature or self.config["temperature"]
        do_sample = do_sample if do_sample is not None else self.config["do_sample"]

        if not self.speculative_enabled:
            return self._generate_batch_baseline(
                prompts, max_tokens, temperature, do_sample, **kwargs
            )

        tokenizer = (
            self.base_lm._tokenizer if hasattr(self.base_lm, "_tokenizer") else None
        )
        if tokenizer is None:
            self.logger.warning(
                "No tokenizer access, falling back to sequential processing"
            )
            return [
                self.generate(prompt, max_tokens, temperature, do_sample, **kwargs)
                for prompt in prompts
            ]

        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        if (
            self.speculative_enabled
            and hasattr(self.draft_lm, "_tokenizer")
            and self.draft_lm._tokenizer is not None
        ):
            draft_tokenizer = self.draft_lm._tokenizer
            if hasattr(draft_tokenizer, "padding_side"):
                draft_tokenizer.padding_side = "left"

        encoded = tokenizer(
            prompts, padding=True, return_tensors="pt", return_attention_mask=True
        )

        if self.device == "cuda" and torch.cuda.is_available():
            batch_input_ids = (
                encoded["input_ids"].pin_memory().to(self.device, non_blocking=True)
            )
        else:
            batch_input_ids = encoded["input_ids"].to(self.device)

        mem_before = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )

        if self.optimization_manager:
            optimization_context = self.optimization_manager.get_optimization_context()
        else:
            optimization_context = (
                amp_context(self.device, self.dtype)
                if self.amp_enabled
                else torch.no_grad()
            )

        if hasattr(self.base_lm, "clear_kv_cache"):
            self.base_lm.clear_kv_cache()
        if hasattr(self.draft_lm, "clear_kv_cache"):
            self.draft_lm.clear_kv_cache()

        kv_cache_enabled = (
            os.getenv("SPECDEC_ENABLE_KV_APPEND", "0") == "1"
            and hasattr(self.base_lm, "supports_kv_append")
            and self.base_lm.supports_kv_append()
        )

        self.kv_cache_manager.set_batch_size(batch_size)

        handlers = self._create_batch_handlers()
        current_input_ids = [batch_input_ids[i].clone() for i in range(batch_size)]
        initial_sequence_lengths = [seq.shape[0] for seq in current_input_ids]
        global_sequence_ids = list(range(batch_size))

        sequence_manager = BatchSequenceManager(
            batch_size=batch_size,
            initial_sequence_lengths=initial_sequence_lengths,
            global_sequence_ids=global_sequence_ids,
        )

        metrics_collector = BatchMetricsCollector()
        metrics_collector.initialize_per_prompt_tracking(batch_size)

        # Initialize streams (used for CUDA async execution)
        draft_stream = None
        verify_stream = None
        cuda_launch_blocking = os.getenv("CUDA_LAUNCH_BLOCKING") == "1"

        if (
            self.device == "cuda"
            and torch.cuda.is_available()
            and not cuda_launch_blocking
        ):
            if (
                hasattr(self.scheduler, "verification_stream")
                and self.scheduler.verification_stream
            ):
                verify_stream = self.scheduler.verification_stream
                draft_stream = torch.cuda.current_stream()
            else:
                draft_stream = torch.cuda.Stream()
                verify_stream = torch.cuda.Stream()

        if kv_cache_enabled:
            self.kv_cache_manager.set_sequence_metadata(
                global_sequence_ids=global_sequence_ids,
                sequence_lengths=initial_sequence_lengths,
            )
            if len(self.kv_cache_manager.base_current_seq_lens) != batch_size:
                self.kv_cache_manager.base_current_seq_lens = (
                    initial_sequence_lengths.copy()
                )
            if len(self.kv_cache_manager.draft_current_seq_lens) != batch_size:
                self.kv_cache_manager.draft_current_seq_lens = (
                    initial_sequence_lengths.copy()
                )

        batch_loop = BatchGenerationLoop(
            draft_handler=handlers["draft"],
            verify_handler=handlers["verify"],
            accept_handler=handlers["accept"],
            rollback_handler=handlers["rollback"],
            kv_handler=handlers["kv"],
            scheduler=self.scheduler,
            policy=self.policy,
            tokenizer=tokenizer,
            device=self.device,
            kv_cache_manager=self.kv_cache_manager,
            logger=self.logger,
        )

        with optimization_context:
            batch_generated_tokens, batch_metrics = batch_loop.run(
                prompts=prompts,
                batch_input_ids=batch_input_ids,
                current_input_ids=current_input_ids,
                sequence_manager=sequence_manager,
                metrics_collector=metrics_collector,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                controller=self.controller,
                kv_cache_enabled=kv_cache_enabled,
                draft_stream=draft_stream,
                verify_stream=verify_stream,
                **kwargs,
            )

        total_time_ms = batch_metrics["total_generation_time_ms"]

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                if draft_stream is not None:
                    draft_stream.synchronize()
                if verify_stream is not None:
                    verify_stream.synchronize()
                if hasattr(self.base_lm, "clear_kv_cache"):
                    self.base_lm.clear_kv_cache()
                if hasattr(self.draft_lm, "clear_kv_cache"):
                    self.draft_lm.clear_kv_cache()
                self.kv_cache_manager.reset()
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                self.logger.warning(f"Error cleaning up CUDA resources: {e}")

        results = []
        per_prompt_metrics = metrics_collector.get_per_prompt_metrics()

        for i, (prompt, generated_tokens) in enumerate(
            zip(prompts, batch_generated_tokens)
        ):
            generated_text = (
                self.base_lm.decode(generated_tokens) if generated_tokens else ""
            )

            prompt_acceptance_rate = batch_metrics["total_accepted"] / max(
                batch_metrics["total_proposed"], 1
            )
            prompt_throughput = (
                len(generated_tokens) / (total_time_ms / 1000.0)
                if total_time_ms > 0
                else 0.0
            )
            prompt_latency_ms = (
                total_time_ms / len(generated_tokens)
                if len(generated_tokens) > 0
                else 0.0
            )

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

            try:
                from kernels import get_kernel_info

                kernel_info = get_kernel_info()
                kv_append_backend = kernel_info.get("kv_append_backend", "unknown")
            except ImportError:
                kv_append_backend = "unavailable"

            results.append(
                {
                    "prompt": prompt,
                    "text": generated_text,
                    "generated_text": generated_text,
                    "generated_tokens": generated_tokens,
                    "num_generated": len(generated_tokens),
                    "batch_index": i,
                    "batch_size": batch_size,
                    "latency_ms": prompt_latency_ms,
                    "total_time_ms": total_time_ms,
                    "tokens_per_sec": prompt_throughput,
                    "throughput_tokens_per_sec": prompt_throughput,
                    "acceptance_rate": prompt_acceptance_rate,
                    "proposed": per_prompt_metrics["proposed"][i],
                    "accepted": per_prompt_metrics["accepted"][i],
                    "draft_avg_ms": avg_draft_time,
                    "verify_avg_ms": avg_verify_time,
                    "batch_metrics": batch_metrics,
                    "kv_append_enabled": False,
                    "kv_append_backend": kv_append_backend,
                    "kv_appended_tokens": 0,
                    "kv_append_time_ms": 0.0,
                }
            )

        mem_after = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )
        mem_used_mb = (mem_after - mem_before) / (1024 * 1024)

        if mem_after > 0:
            self.logger.info(
                f"Batch generation complete: {batch_size} prompts, "
                f"GPU memory used: {mem_used_mb:.2f} MB"
            )

        return results
