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

logger = logging.getLogger(__name__)


# filter_kv_cache_safe removed - use SafeKVCacheManager instead


# Import optimization modules (conditional to avoid import errors during development)
try:
    from benchmarks.profiler import create_profiler
    from metrics.structured_profiler import create_structured_profiler
    from optimization import (
        amp_context,
        create_optimization_manager,
        create_optimized_tokenizer,
        select_device_dtype,
    )
    from scheduler import create_speculative_scheduler
except ImportError:
    # Fallback for development - create dummy functions
    def create_optimization_manager(*args, **kwargs):
        return None

    def create_optimized_tokenizer(*args, **kwargs):
        return None

    def create_profiler(*args, **kwargs):
        return None

    def create_structured_profiler(*args, **kwargs):
        # Return dummy structured profiler for development
        class DummyProfiler:
            enable_profiling = False

            def record_step(self, *args, **kwargs):
                pass

            def record_kv_append_time(self, *args, **kwargs):
                pass

        return DummyProfiler()

    def create_speculative_scheduler(*args, **kwargs):
        return None

    def select_device_dtype(device="auto"):
        return device, torch.float32, False

    def amp_context(device, dtype):
        return torch.no_grad()


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

        # Initialize centralized KV cache manager
        self.kv_cache_manager = SafeKVCacheManager(device=str(self.device))

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

        # Initialize policy and controller
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

                if hasattr(self.draft_lm, "optimize"):
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

    def _create_draft_model(self) -> LanguageModel:
        """Create the draft language model based on implementation."""
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

                # Log acceptance details
                proposed_count = draft_tokens.shape[1]
                rejected_count = proposed_count - accepted_len
                print(
                    f"[SCHED] Step {step} | "
                    f"K={proposed_count} | "
                    f"accepted={accepted_len} | "
                    f"rejected={rejected_count} | "
                    f"draft={draft_time_ms:.1f}ms | "
                    f"verify={verify_time_ms:.1f}ms",
                    flush=True,
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
                    # to base model cache
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
                                    # Slice to accepted length
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

        batch_size = len(prompts)
        self.logger.info(f"Starting batched generation for {batch_size} prompts")
        print(
            f"[BATCH] Starting batch processing: {batch_size} prompts, max_tokens={max_tokens}",
            flush=True,
        )

        # Use provided parameters or fall back to config
        max_tokens = max_tokens or self.config["max_new_tokens"]
        temperature = temperature or self.config["temperature"]
        do_sample = do_sample if do_sample is not None else self.config["do_sample"]

        # Model init diagnostics (only print once)
        if not hasattr(self, "_batch_init_printed"):
            print(
                f"[BATCH] Base model: {self.config.get('base_model', 'unknown')}",
                flush=True,
            )
            print(
                f"[BATCH] Draft model: {self.config.get('draft_model', 'unknown')}",
                flush=True,
            )
            print(f"[BATCH] Device: {self.device}, Dtype: {self.dtype}", flush=True)
            if self.device == "cuda" and torch.cuda.is_available():
                print(f"[BATCH] GPU: {torch.cuda.get_device_name(0)}", flush=True)
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
            print(
                "[BATCH] Warning: No tokenizer access, falling back to sequential",
                flush=True,
            )
            return [
                self.generate(prompt, max_tokens, temperature, do_sample, **kwargs)
                for prompt in prompts
            ]

        # Tokenizer alignment check - ensure both models use same tokenizer
        if (
            hasattr(self.draft_lm, "_tokenizer")
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
                        print(
                            f"[CHECK] Tokenizer alignment OK - vocab_size={base_vocab_size}",
                            flush=True,
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
                                    print(
                                        f"[CHECK] Tokenizer overlap sanity: {overlap:.2f} "
                                        f"(expected ~1.0 for prefix match)",
                                        flush=True,
                                    )
                                self._tokenizer_warmup_done = True
                            except Exception as e:
                                print(
                                    f"[CHECK] Tokenizer warmup test failed: {e}",
                                    flush=True,
                                )

        # Tokenize with padding (pre-tokenize once, reuse across iterations)
        print(f"[BATCH] Tokenizing {batch_size} prompts with padding...", flush=True)
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
        print(
            f"[BATCH] Tokenized batch shape: {batch_input_ids.shape} "
            f"(pinned={'yes' if self.device == 'cuda' and torch.cuda.is_available() else 'no'})",
            flush=True,
        )

        # Track memory before batch
        mem_before = (
            torch.cuda.memory_allocated()
            if self.device == "cuda" and torch.cuda.is_available()
            else 0
        )
        if mem_before > 0:
            print(
                f"[BATCH] GPU memory before: {mem_before / (1024**2):.2f} MB",
                flush=True,
            )

        # VECTORIZED BATCH PROCESSING: Process all prompts together in parallel
        # This enables true GPU parallelism with batched tensor operations
        print(
            f"[BATCH] Starting vectorized speculative decoding for {batch_size} prompts",
            flush=True,
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
            print(
                "[BATCH] KV cache append enabled - using SafeKVCacheManager",
                flush=True,
            )
        else:
            if kv_cache_enabled_env and not kv_cache_supported:
                print(
                    "[BATCH] Warning: SPECDEC_ENABLE_KV_APPEND=1 but model doesn't support KV append",
                    flush=True,
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

            step = 0
            generation_start = time.time()

            # Create CUDA streams for draft/verify overlap
            draft_stream = None
            verify_stream = None
            if self.device == "cuda" and torch.cuda.is_available():
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

            while step < max_tokens:
                step += 1
                active_count = sum(batch_active)

                # Debug print when active_count drops
                if step > 1 and active_count < batch_size:
                    finished_count = batch_size - active_count
                    print(
                        "[INFO] Step {}: {} sequence(s) finished, {} still active".format(
                            step, finished_count, active_count
                        ),
                        flush=True,
                    )

                if active_count == 0:
                    print(
                        "[INFO] Step {}: All sequences finished, exiting loop".format(
                            step
                        ),
                        flush=True,
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
                    print(
                        f"[WARNING] Step {step}: K={k} <= 0, skipping generation",
                        flush=True,
                    )
                    break

                # Filter to active prompts only
                active_indices = [i for i, active in enumerate(batch_active) if active]
                if not active_indices:
                    break

                # Update KV cache manager with active indices
                # Use relative indices (0 to len-1) for KV cache, not absolute batch indices
                relative_indices = list(range(len(active_indices)))
                self.kv_cache_manager.set_active_indices(relative_indices)
                active_indices_tensor = torch.tensor(
                    active_indices, dtype=torch.long, device=self.device
                )
                relative_indices_tensor = torch.tensor(
                    relative_indices, dtype=torch.long, device=self.device
                )

                # Create batched input for active prompts only
                active_seqs = [current_input_ids[i] for i in active_indices]
                if len(active_seqs) == 0:
                    break

                # Single centralized validation point - efficient GPU operation
                # Note: We validate before padding since sequences may have different lengths
                # Validation happens on individual sequences (already on GPU) to avoid padding overhead
                vocab_size = get_vocab_size(self.base_lm)
                if vocab_size is not None and len(active_seqs) > 0:
                    # Validate each sequence in-place on GPU (minimal CPU overhead)
                    # This is more efficient than padding first then validating
                    for i, seq in enumerate(active_seqs):
                        active_seqs[i] = validate_and_clamp_tokens(
                            seq, vocab_size, f"seq_{i}"
                        )

                # Find max sequence length in active batch (optimized: single pass)
                original_lengths = []
                max_seq_len = 0
                for seq in active_seqs:
                    seq_len = seq.shape[0]
                    original_lengths.append(seq_len)
                    max_seq_len = max(max_seq_len, seq_len)

                # Pad all sequences to max length for batching
                vocab_size = get_vocab_size(self.base_lm)
                pad_value = (
                    tokenizer.pad_token_id
                    if tokenizer is not None and tokenizer.pad_token_id is not None
                    else 0
                )
                # Ensure pad_value is valid
                if vocab_size is not None and (
                    pad_value >= vocab_size or pad_value < 0
                ):
                    pad_value = 0 if vocab_size > 0 else 0

                # Pad sequences efficiently
                padded_seqs: List[Optional[torch.Tensor]] = [None] * len(active_seqs)
                for i, seq in enumerate(active_seqs):
                    if seq.shape[0] < max_seq_len:
                        pad_length = max_seq_len - seq.shape[0]
                        padding = torch.full(
                            (pad_length,),
                            pad_value,
                            dtype=seq.dtype,
                            device=seq.device,
                        )
                        seq_padded = torch.cat([seq, padding], dim=0)
                        padded_seqs[i] = seq_padded
                    else:
                        padded_seqs[i] = seq

                # Stack into batch tensor
                active_input_ids = torch.stack(
                    padded_seqs, dim=0
                )  # [active_count, max_seq_len]

                # Single final validation - efficient GPU operation
                if vocab_size is not None:
                    active_input_ids = validate_and_clamp_tokens(
                        active_input_ids, vocab_size, "batch_input"
                    )

                # Create attention mask to ignore padding tokens
                active_attention_mask = torch.ones(
                    (len(active_indices), max_seq_len),
                    dtype=torch.long,
                    device=self.device,
                )
                for i, orig_len in enumerate(original_lengths):
                    if orig_len < max_seq_len:
                        # Mask out padding tokens
                        active_attention_mask[i, orig_len:] = 0

                # Debug: validate input_ids are not all padding
                non_pad_tokens = (active_input_ids != pad_value).sum().item()
                total_tokens = active_input_ids.numel()

                if step == 1:
                    print(
                        f"[BATCH] Batched active_input_ids shape: {active_input_ids.shape} "
                        f"(active={len(active_indices)}, max_len={max_seq_len}, "
                        f"original_lens={original_lengths}, non_pad={non_pad_tokens}/{total_tokens})",
                        flush=True,
                    )
                    # Decode and print first prompt for verification
                    if tokenizer is not None and active_input_ids.shape[0] > 0:
                        first_prompt_tokens = active_input_ids[0]
                        # Remove padding before decoding
                        first_prompt_non_pad = first_prompt_tokens[
                            first_prompt_tokens != pad_value
                        ]
                        if len(first_prompt_non_pad) > 0:
                            try:
                                decoded_text = tokenizer.decode(first_prompt_non_pad)
                                print(
                                    f"[DEBUG] First prompt decoded: {decoded_text[:100]}...",
                                    flush=True,
                                )
                            except Exception as e:
                                print(
                                    f"[DEBUG] Failed to decode first prompt: {e}",
                                    flush=True,
                                )

                if non_pad_tokens == 0:
                    print(
                        f"[WARNING] Step {step}: All tokens are padding, skipping generation",
                        flush=True,
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

                # Debug: log before draft execution (only first step to reduce CPU overhead)
                if step == 1:
                    print(
                        f"[DEBUG] Before draft - input shape: {active_input_ids.shape}, "
                        f"K={k}, device={active_input_ids.device}",
                        flush=True,
                    )

                # Apply temperature stabilization for draft model
                # Use lower effective temperature to improve acceptance rate
                draft_temperature = (
                    max(temperature / 1.5, 0.1) if temperature is not None else 0.7
                )

                # Prepare past_key_values for draft model using centralized KV cache manager
                draft_past_kv = None
                if kv_cache_enabled:
                    # Use relative indices for KV cache retrieval (0 to active_count-1)
                    draft_past_kv = self.kv_cache_manager.get_draft_past_kv(
                        relative_indices_tensor
                    )

                # Single validation point (already done above, but double-check before model call)
                draft_vocab_size = get_vocab_size(self.draft_lm)
                if draft_vocab_size is not None:
                    active_input_ids = validate_and_clamp_tokens(
                        active_input_ids, draft_vocab_size, "draft_input"
                    )

                if draft_stream is not None:
                    with torch.cuda.stream(draft_stream):
                        try:
                            draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                                active_input_ids,
                                max_new_tokens=k,
                                temperature=draft_temperature,  # Lower temperature for draft
                                do_sample=False,  # Use greedy for draft to maximize acceptance
                                stream=draft_stream,
                                past_key_values=draft_past_kv,
                                **kwargs,
                            )
                        except RuntimeError as e:
                            # Catch CUDA errors and provide better diagnostics
                            if "indexSelectLargeIndex" in str(
                                e
                            ) or "device-side assert" in str(e):
                                print(
                                    f"[CRITICAL ERROR] Step {step}: CUDA embedding error in draft model!\n"
                                    f"active_input_ids.shape={active_input_ids.shape}\n"
                                    f"active_input_ids (min/max unavailable - tensor corrupted)\n"
                                    f"vocab_size={getattr(self.draft_lm._model.config, 'vocab_size', 'unknown') if hasattr(self.draft_lm, '_model') else 'unknown'}\n"
                                    f"Error: {e}",
                                    flush=True,
                                )
                            raise
                    if draft_end_event is not None:
                        draft_end_event.record(draft_stream)
                else:
                    draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                        active_input_ids,
                        max_new_tokens=k,
                        temperature=draft_temperature,  # Lower temperature for draft
                        do_sample=False,  # Use greedy for draft to maximize acceptance
                        past_key_values=draft_past_kv,
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
                    print(
                        f"[ERROR] Step {step}: Draft model returned empty tokens! "
                        f"Shape: {draft_tokens.shape}",
                        flush=True,
                    )
                    break

                # Debug: decode draft tokens (only first step to reduce CPU overhead)
                if step == 1 and tokenizer is not None:
                    try:
                        draft_text = tokenizer.decode(draft_tokens[0])
                        print(
                            f"[DEBUG] Draft tokens decoded (first prompt): {draft_text[:100]}...",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[DEBUG] Failed to decode draft tokens: {e}", flush=True)

                # Calculate draft time using CUDA events if available, otherwise wall-clock
                if draft_start_event is not None and draft_end_event is not None:
                    draft_end_event.synchronize()
                    draft_time_ms = draft_start_event.elapsed_time(draft_end_event)
                else:
                    draft_time_ms = (time.time() - draft_start_wall) * 1000

                # Debug: log draft outputs shape (reduced frequency to reduce CPU overhead)
                if step == 1 or step % 16 == 0:
                    print(
                        f"[DEBUG] Draft execution - tokens shape: {draft_tokens.shape}, "
                        f"logits shape: {draft_logits.shape}, "
                        f"time: {draft_time_ms:.2f}ms, "
                        f"proposed_tokens: {draft_tokens.shape[0] * draft_tokens.shape[1]}",
                        flush=True,
                    )
                if step == 1:
                    # Log sample token IDs (only first step)
                    print(
                        f"[DEBUG] Sample draft token IDs (first batch): "
                        f"{draft_tokens[0, :min(5, draft_tokens.shape[1])].tolist()}",
                        flush=True,
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
                        base_tokens, base_logits, verify_info = (
                            self.scheduler.schedule_verification(
                                self.base_lm,
                                dummy_draft_tokens,  # Shape only, not actual draft tokens
                                active_input_ids,
                                temperature=temperature,
                                do_sample=do_sample,
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

                    # Calculate verify time using CUDA events if available
                    if verify_start_event is not None and verify_end_event is not None:
                        verify_time_ms = verify_start_event.elapsed_time(
                            verify_end_event
                        )
                    else:
                        verify_time_ms = (time.time() - verify_start_wall) * 1000
                elif verify_stream is not None and draft_stream is not None:
                    # Manual stream management if scheduler not available
                    # Prepare past_key_values for base model using centralized manager
                    base_past_kv = None
                    if kv_cache_enabled:
                        # Use relative indices for KV cache retrieval (0 to active_count-1)
                        base_past_kv = self.kv_cache_manager.get_base_past_kv(
                            relative_indices_tensor
                        )

                    with torch.cuda.stream(verify_stream):
                        if verify_start_event is not None:
                            verify_start_event.record(verify_stream)
                        # For verification, always use greedy (argmax) to ensure deterministic matching
                        # This ensures base_logits.argmax() matches the tokens we compare
                        try:
                            base_tokens, base_logits = self.base_lm.generate_tokens(
                                active_input_ids,
                                max_new_tokens=k,  # Use k directly, not draft_tokens.shape[1]
                                temperature=1.0,  # Temperature=1.0 for deterministic argmax
                                do_sample=False,  # Always use greedy for verification
                                stream=verify_stream,
                                past_key_values=base_past_kv,
                                **kwargs,
                            )
                        except RuntimeError as e:
                            # Catch CUDA errors and provide better diagnostics
                            if "indexSelectLargeIndex" in str(
                                e
                            ) or "device-side assert" in str(e):
                                print(
                                    f"[CRITICAL ERROR] Step {step}: CUDA embedding error in base model (scheduler path)!\n"
                                    f"active_input_ids.shape={active_input_ids.shape}\n"
                                    f"active_input_ids (min/max unavailable - tensor corrupted)\n"
                                    f"vocab_size={getattr(self.base_lm._model.config, 'vocab_size', 'unknown') if hasattr(self.base_lm, '_model') else 'unknown'}\n"
                                    f"base_past_kv is None={base_past_kv is None}\n"
                                    f"Error: {e}",
                                    flush=True,
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
                        if base_new_kv is not None:
                            if isinstance(base_new_kv, tuple):
                                self.kv_cache_manager.update_base_cache(
                                    base_new_kv, active_indices
                                )
                            else:
                                if len(base_new_kv) > 0:
                                    self.kv_cache_manager.update_base_cache(
                                        tuple(base_new_kv), active_indices
                                    )

                    # Synchronize both streams
                    if draft_end_event is not None:
                        draft_end_event.synchronize()
                    if verify_end_event is not None:
                        verify_end_event.synchronize()

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

                    # Prepare past_key_values for base model using centralized manager
                    base_past_kv = None
                    if kv_cache_enabled:
                        # Use relative indices for KV cache retrieval (0 to active_count-1)
                        base_past_kv = self.kv_cache_manager.get_base_past_kv(
                            relative_indices_tensor
                        )

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
                                    base_new_kv, active_indices
                                )
                            else:
                                if len(base_new_kv) > 0:
                                    self.kv_cache_manager.update_base_cache(
                                        tuple(base_new_kv), active_indices
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
                    if overlap_saved_ms > 1.0:  # Only log if significant overlap
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

                # Debug: log batch metrics after each step (reduced frequency to reduce CPU overhead)
                if step == 1 or step % 16 == 0:
                    print(
                        f"[DEBUG] Step {step} metrics - "
                        f"proposed: {batch_metrics['total_proposed']}, "
                        f"accepted: {batch_metrics['total_accepted']}, "
                        f"draft_time_total: {batch_metrics['total_draft_time_ms']:.2f}ms, "
                        f"verify_time_total: {batch_metrics['total_verification_time_ms']:.2f}ms",
                        flush=True,
                    )

                # Step 3: Apply acceptance policy in batch
                accepted_lengths = []
                accepted_tokens_list = []

                # Debug: log verify outputs shape (only first step and every 16 steps to reduce CPU overhead)
                if step == 1 or step % 16 == 0:
                    print(
                        f"[DEBUG] Verify execution - base_tokens shape: {base_tokens.shape}, "
                        f"base_logits shape: {base_logits.shape}, "
                        f"verify_time: {verify_time_ms:.2f}ms",
                        flush=True,
                    )
                if step == 1:
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

                # Process each active prompt's acceptance
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
                    # Debug: log tokens before policy (with argmax for verification)
                    # Minimize CPU transfers - only when debugging
                    if step <= 2 and idx_in_active == 0:
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
                            matches = sum(
                                1
                                for i in range(
                                    min(
                                        len(draft_tokens_sample),
                                        len(base_pred_from_logits),
                                    )
                                )
                                if draft_tokens_sample[i] == base_pred_from_logits[i]
                            )
                            overlap_ratio = matches / max(
                                len(draft_tokens_sample), len(base_pred_from_logits)
                            )
                            print(
                                f"[DEBUG] Token overlap ratio: {overlap_ratio:.2f} ({matches}/{min(len(draft_tokens_sample), len(base_pred_from_logits))} match)",
                                flush=True,
                            )

                    accepted_len, policy_info = self.policy.accept_tokens(
                        prompt_draft_tokens,
                        prompt_base_tokens,
                        prompt_draft_logits,
                        prompt_base_logits,
                    )

                    # Debug: log policy result (reduced frequency unless debug mode enabled)
                    debug_accept = os.getenv("SPECDEC_DEBUG_ACCEPT", "0") == "1"
                    if (step <= 2 or debug_accept) and (step % 8 == 0 or step <= 2):
                        proposed_count = prompt_draft_tokens.shape[1]
                        accept_rate = accepted_len / max(proposed_count, 1)
                        print(
                            f"[DEBUG] Step {step} | Proposed={proposed_count} | Accepted={accepted_len} | "
                            f"AcceptRate={accept_rate:.2%} | Policy={policy_info.get('verify_backend', 'unknown')}",
                            flush=True,
                        )

                    # Get accepted tokens - ensure we always have something
                    # Extract accepted tokens with centralized validation
                    accepted_tokens = []
                    if accepted_len > 0:
                        accepted_tokens_tensor = prompt_draft_tokens[0, :accepted_len]
                        # Validate before CPU transfer
                        base_vocab_size = get_vocab_size(self.base_lm)
                        if base_vocab_size is not None:
                            accepted_tokens_tensor = validate_and_clamp_tokens(
                                accepted_tokens_tensor,
                                base_vocab_size,
                                "accepted_tokens",
                            )
                        accepted_tokens = accepted_tokens_tensor.cpu().tolist()
                        batch_generated_tokens[global_idx].extend(accepted_tokens)
                        accepted_tokens_list.append(accepted_tokens)
                    else:
                        # Rejected all - accept first base token as fallback
                        first_base_tensor = prompt_base_tokens[0, 0:1]
                        # Validate with centralized validation
                        base_vocab_size = get_vocab_size(self.base_lm)
                        if base_vocab_size is not None:
                            first_base_tensor = validate_and_clamp_tokens(
                                first_base_tensor, base_vocab_size, "fallback_base"
                            )
                        accepted_tokens = first_base_tensor.cpu().tolist()
                        batch_generated_tokens[global_idx].extend(accepted_tokens)
                        accepted_tokens_list.append(accepted_tokens)
                        accepted_len = 1
                        if step <= 3 or idx_in_active == 0:
                            print(
                                f"[DEBUG] No draft tokens accepted - accepting first base token: {accepted_tokens}",
                                flush=True,
                            )

                    # Debug: log acceptance per prompt (reduced frequency to reduce CPU overhead)
                    debug_accept = os.getenv("SPECDEC_DEBUG_ACCEPT", "0") == "1"
                    if (step <= 2 or debug_accept) and (step % 8 == 0 or step <= 2):
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

                        # Validate current sequence with centralized validation
                        if base_vocab_size is not None:
                            current_seq = validate_and_clamp_tokens(
                                current_seq,
                                base_vocab_size,
                                f"current_seq_{global_idx}",
                            )

                        # Debug: log before/after append (only first step to reduce CPU overhead)
                        if step == 1:
                            print(
                                f"[DEBUG] Sequence update - prompt {global_idx}: "
                                f"before_len={current_seq.shape[0]}, "
                                f"accepted_len={len(accepted_tokens)}, "
                                f"after_len={current_seq.shape[0] + len(accepted_tokens)}",
                                flush=True,
                            )

                        # Concatenate along sequence dimension (both are 1D)
                        updated_seq = torch.cat(
                            [current_seq, accepted_tokens_tensor], dim=0
                        )

                        # Validate after concatenation with centralized validation
                        base_vocab_size = get_vocab_size(self.base_lm)
                        if base_vocab_size is not None:
                            updated_seq = validate_and_clamp_tokens(
                                updated_seq,
                                base_vocab_size,
                                f"updated_seq_{global_idx}",
                            )

                        # Update sequence (keep as 1D list item, no padding)
                        current_input_ids[global_idx] = updated_seq
                    else:
                        # No tokens accepted - log warning but continue with next base token
                        print(
                            f"[WARNING] Step {step}, prompt {global_idx}: No tokens to append!",
                            flush=True,
                        )

                    # Check if this prompt is done
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
                    # Check if prompt is done
                    if len(batch_generated_tokens[global_idx]) >= max_tokens:
                        batch_active[global_idx] = False
                    elif (
                        len(accepted_tokens) > 0 and accepted_tokens[-1] == eos_token_id
                    ):
                        batch_active[global_idx] = False

                batch_metrics["total_steps"] += 1

                # Log batch progress with accurate timing
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

            # Convert batched results to per-prompt dictionaries
            results = []
            for i, (prompt, generated_tokens) in enumerate(
                zip(prompts, batch_generated_tokens)
            ):
                # Decode generated tokens
                if generated_tokens:
                    generated_text = self.base_lm.decode(
                        torch.tensor(generated_tokens, device=self.device).unsqueeze(0)
                    )
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
