"""
Hugging Face Model Wrappers for Speculative Decoding

Provides HFWrapper that wraps Hugging Face models with safe defaults
for tiny models only (smoke runs) and MPS memory hygiene.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interfaces import LanguageModel
from .kv_types import KVCache


class HFWrapper(LanguageModel):
    """Wrapper for Hugging Face models with memory-safe defaults."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        max_memory_mb: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize the Hugging Face wrapper.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ("auto", "cpu", "mps", "cuda")
            torch_dtype: PyTorch data type for the model (auto-selected if None)
            max_memory_mb: Maximum memory usage in MB (for safety)
            tokenizer: Shared tokenizer instance (to avoid duplicate RAM)
        """
        self.logger = logging.getLogger(__name__)
        self._model_name = model_name
        self._device = self._select_device(device)
        self._torch_dtype = torch_dtype or self._select_dtype()
        self._max_memory_mb = max_memory_mb
        self._tokenizer = tokenizer

        # KV cache management
        self._kv_cache: Optional[KVCache] = None
        self._kv_append_enabled = os.getenv(
            "SPECDEC_ENABLE_KV_APPEND", "1"
        ).lower() in ("1", "true", "yes")

        # Load model and tokenizer
        self._load_model()

    def _select_device(self, device: str) -> str:
        """Select the best available device with memory considerations."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _select_dtype(self) -> torch.dtype:
        """Select appropriate dtype based on device."""
        if self._device in ["cuda", "mps"]:
            return torch.float16
        else:
            return torch.float32

    def _load_model(self) -> None:
        """Load the model and tokenizer with memory safety."""
        try:
            self.logger.info(f"Loading HF model: {self._model_name}")

            # Load tokenizer (shared if provided)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                if self._tokenizer.pad_token is None:  # type: ignore
                    self._tokenizer.pad_token = (  # type: ignore
                        self._tokenizer.eos_token  # type: ignore
                    )
                self._tokenizer.padding_side = "left"  # type: ignore

            # Load model with memory considerations
            model_kwargs = {
                "torch_dtype": self._torch_dtype,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            }

            # Only use device_map if accelerate is available
            try:
                # import accelerate  # Unused import

                if self._device != "cpu":
                    model_kwargs["device_map"] = {"": 0}
            except ImportError:
                # Fallback: load to CPU first, then move to device
                pass

            # Add memory constraints if specified
            if self._max_memory_mb:
                model_kwargs["max_memory"] = {0: f"{self._max_memory_mb}MB"}

            self._model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                self._model_name, **model_kwargs
            )

            # Move to device if needed
            # (device_map handles this for GPU if accelerate is available)
            if self._device != "auto" and self._device != "cpu":
                try:
                    # # import accelerate  # Unused import  # Unused import

                    # If accelerate is available and device_map was used,
                    # model is already on device
                    if "device_map" not in model_kwargs or not hasattr(
                        self._model, "hf_device_map"
                    ):
                        self._model = self._model.to(self._device)  # type: ignore
                except ImportError:
                    # No accelerate, manually move to device
                    self._model = self._model.to(self._device)  # type: ignore

            self._model.eval()

            self.logger.info(f"HF model loaded on device: {self._device}")

        except Exception as e:
            self.logger.error(f"Failed to load HF model: {e}")
            raise

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        stream: Optional[torch.cuda.Stream] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens using the Hugging Face model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            stream: Optional CUDA stream for async execution
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_ids, logits)
        """
        try:
            # CRITICAL: Validate input_ids BEFORE any model operations
            # This is the final safety check before embedding layer
            if hasattr(self._model, "config"):
                vocab_size = getattr(self._model.config, "vocab_size", None)
                if vocab_size is not None:
                    if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                        # Get min/max safely - move to CPU first to avoid CUDA errors
                        try:
                            max_token = input_ids.cpu().max().item()
                            min_token = input_ids.cpu().min().item()
                        except Exception:
                            max_token = vocab_size  # Assume worst case
                            min_token = -1
                        invalid_count = (
                            ((input_ids >= vocab_size) | (input_ids < 0)).sum().item()
                        )
                        self.logger.error(
                            "[HF-WRAPPER] Invalid token indices detected: "
                            "min=%d, max=%d, vocab_size=%d, invalid_count=%d/%d",
                            min_token,
                            max_token,
                            vocab_size,
                            invalid_count,
                            input_ids.numel(),
                        )
                        print(
                            f"[ERROR] Invalid tokens in generate_tokens: "
                            f"min={min_token}, max={max_token}, vocab={vocab_size}, "
                            f"invalid={invalid_count}/{input_ids.numel()}",
                            flush=True,
                        )
                        # Clamp invalid tokens to valid range
                        input_ids = input_ids.clamp(min=0, max=vocab_size - 1)

            # Use async generation if stream is provided and device is CUDA
            if stream is not None and self._device == "cuda":
                return self._generate_tokens_async(
                    input_ids,
                    max_new_tokens,
                    temperature,
                    do_sample,
                    stream,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            with torch.no_grad():
                # Move input to device if needed
                if input_ids.device != torch.device(self._device):
                    input_ids = input_ids.to(self._device)

                # CRITICAL: Re-validate after device transfer (tensors can be corrupted)
                if hasattr(self._model, "config"):
                    vocab_size = getattr(self._model.config, "vocab_size", None)
                    if vocab_size is not None:
                        if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                            # Get min/max safely - move to CPU first to avoid CUDA errors
                            try:
                                max_token = input_ids.cpu().max().item()
                                min_token = input_ids.cpu().min().item()
                            except Exception:
                                max_token = vocab_size  # Assume worst case
                                min_token = -1
                            print(
                                f"[ERROR] Invalid tokens after device transfer: "
                                f"min={min_token}, max={max_token}, "
                                f"vocab={vocab_size}",
                                flush=True,
                            )
                            # Clamp invalid tokens to valid range
                            input_ids = input_ids.clamp(min=0, max=vocab_size - 1)

                # Use KV-aware generation if supported and enabled
                if self.supports_kv_append() and self._kv_cache is not None:
                    return self._generate_with_kv_cache(
                        input_ids, max_new_tokens, temperature, do_sample, **kwargs
                    )

                # Standard generation without KV cache
                # CRITICAL: Final validation before model.generate()
                if hasattr(self._model, "config"):
                    vocab_size = getattr(self._model.config, "vocab_size", None)
                    if vocab_size is not None:
                        if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                            # Get min/max safely - move to CPU first to avoid CUDA errors
                            try:
                                max_token = input_ids.cpu().max().item()
                                min_token = input_ids.cpu().min().item()
                            except Exception:
                                max_token = vocab_size  # Assume worst case
                                min_token = -1
                            print(
                                f"[ERROR] Invalid tokens before model.generate(): "
                                f"min={min_token}, max={max_token}, "
                                f"vocab={vocab_size}",
                                flush=True,
                            )
                            # Clamp invalid tokens
                            input_ids = input_ids.clamp(min=0, max=vocab_size - 1)

                # Create attention mask for proper padding handling
                attention_mask = torch.ones_like(input_ids)

                # Generate tokens
                outputs = self._model.generate(  # type: ignore
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id,  # type: ignore
                    eos_token_id=self._tokenizer.eos_token_id,  # type: ignore
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs,
                )

                # Extract generated token IDs (excluding input)
                generated_ids = outputs.sequences[  # type: ignore[union-attr]
                    :, input_ids.shape[1] :
                ]

                # Extract logits for the generated tokens
                if outputs.scores:  # type: ignore
                    logits = torch.stack(outputs.scores, dim=1)  # type: ignore
                else:
                    # Fallback: get logits from the last layer
                    with torch.no_grad():
                        last_hidden_states = self._model(input_ids).logits
                        logits = last_hidden_states[:, -max_new_tokens:, :]

                # Capture KV cache if enabled (from the generation)
                if (
                    self.supports_kv_append()
                    and hasattr(outputs, "past_key_values")
                    and outputs.past_key_values is not None  # type: ignore
                ):
                    self._last_generated_kv = KVCache.from_hf_output(
                        outputs.past_key_values  # type: ignore
                    )
                else:
                    self._last_generated_kv = None

                return generated_ids, logits

        except Exception as e:
            self.logger.error(f"Failed to generate tokens: {e}")
            raise

    def _generate_tokens_async(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stream: torch.cuda.Stream,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens asynchronously using manual forward loop in CUDA stream.
        This enables true async overlap between draft and verification.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            stream: CUDA stream for async execution
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_ids, logits)
        """
        # CRITICAL: Validate input_ids BEFORE async execution
        # This prevents invalid tokens from reaching the embedding layer during async operations
        if hasattr(self._model, "config"):
            vocab_size = getattr(self._model.config, "vocab_size", None)
            if vocab_size is not None:
                if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                    # Get min/max safely - move to CPU first to avoid CUDA errors
                    try:
                        max_token = input_ids.cpu().max().item()
                        min_token = input_ids.cpu().min().item()
                    except Exception:
                        max_token = vocab_size  # Assume worst case
                        min_token = -1
                    invalid_count = (
                        ((input_ids >= vocab_size) | (input_ids < 0)).sum().item()
                    )
                    self.logger.error(
                        "[HF-WRAPPER ASYNC] Invalid token indices: "
                        "min=%d, max=%d, vocab_size=%d, invalid_count=%d/%d",
                        min_token,
                        max_token,
                        vocab_size,
                        invalid_count,
                        input_ids.numel(),
                    )
                    print(
                        f"[ERROR] Invalid tokens in _generate_tokens_async: "
                        f"min={min_token}, max={max_token}, "
                        f"vocab={vocab_size}, invalid={invalid_count}/"
                        f"{input_ids.numel()}",
                        flush=True,
                    )
                    # Clamp invalid tokens to valid range
                    input_ids = input_ids.clamp(min=0, max=vocab_size - 1)

        with torch.cuda.stream(stream):
            with torch.no_grad():
                # Move input to device if needed
                if input_ids.device != torch.device(self._device):
                    input_ids = input_ids.to(self._device)

                # CRITICAL: Re-validate after device transfer in async context
                if hasattr(self._model, "config"):
                    vocab_size = getattr(self._model.config, "vocab_size", None)
                    if vocab_size is not None:
                        if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                            # Get min/max safely - move to CPU first to avoid CUDA errors
                            try:
                                max_token = input_ids.cpu().max().item()
                                min_token = input_ids.cpu().min().item()
                            except Exception:
                                max_token = vocab_size  # Assume worst case
                                min_token = -1
                            print(
                                f"[ERROR] Invalid tokens after device transfer in async: "
                                f"min={min_token}, max={max_token}, vocab={vocab_size}",
                                flush=True,
                            )
                            # Clamp invalid tokens
                            input_ids = input_ids.clamp(min=0, max=vocab_size - 1)

                batch_size, seq_len = input_ids.shape
                generated_tokens = []
                generated_logits = []

                current_input = input_ids
                current_past_kv = past_key_values
                last_past_kv = None  # Track KV cache for return

                for step in range(max_new_tokens):
                    # CRITICAL: Synchronize stream before validation to ensure all previous ops complete
                    if stream is not None:
                        torch.cuda.synchronize()

                    # CRITICAL: Validate current_input before EACH forward pass
                    # This is the absolute last check before embedding lookup in async loop
                    if hasattr(self._model, "config"):
                        vocab_size = getattr(self._model.config, "vocab_size", None)
                        if vocab_size is not None:
                            if (current_input >= vocab_size).any() or (
                                current_input < 0
                            ).any():
                                # Get min/max safely - move to CPU first to avoid CUDA errors
                                try:
                                    max_token = current_input.cpu().max().item()
                                    min_token = current_input.cpu().min().item()
                                except Exception:
                                    max_token = vocab_size  # Assume worst case
                                    min_token = -1
                                invalid_count = (
                                    (
                                        (current_input >= vocab_size)
                                        | (current_input < 0)
                                    )
                                    .sum()
                                    .item()
                                )
                                print(
                                    f"[CRITICAL ERROR] Async loop step {step}: Invalid tokens before forward! "
                                    f"min={min_token}, max={max_token}, vocab={vocab_size}, "
                                    f"invalid={invalid_count}/{current_input.numel()}",
                                    flush=True,
                                )
                                # Clamp invalid tokens
                                current_input = current_input.clamp(
                                    min=0, max=vocab_size - 1
                                )

                    # Forward pass with past_key_values if available
                    if current_past_kv is not None:
                        # Only pass new tokens (not cached ones)
                        # CRITICAL: Final validation RIGHT before model forward in async loop
                        # This is the absolute last check - tokens must be valid here
                        if hasattr(self._model, "config"):
                            vocab_size = getattr(self._model.config, "vocab_size", None)
                            if vocab_size is not None:
                                # Check for invalid tokens using safe operations
                                has_invalid = (current_input >= vocab_size).any() or (
                                    current_input < 0
                                ).any()
                                if has_invalid:
                                    # Calculate invalid count safely
                                    invalid_count = (
                                        (
                                            (current_input >= vocab_size)
                                            | (current_input < 0)
                                        )
                                        .sum()
                                        .item()
                                    )

                                    # Get min/max safely - move to CPU first to avoid CUDA errors
                                    try:
                                        max_token = current_input.cpu().max().item()
                                        min_token = current_input.cpu().min().item()
                                    except Exception as cpu_err:
                                        # If even CPU fails, tensor is severely corrupted
                                        print(
                                            f"[CRITICAL ERROR] Async loop step {step}: Cannot read tensor values! "
                                            f"Error: {cpu_err}",
                                            flush=True,
                                        )
                                        max_token = vocab_size  # Assume worst case
                                        min_token = -1

                                    print(
                                        f"[CRITICAL ERROR] Async loop step {step}: Invalid tokens RIGHT before model forward! "
                                        f"min={min_token}, max={max_token}, vocab={vocab_size}, "
                                        f"invalid={invalid_count}/{current_input.numel()}\n"
                                        f"current_input.shape={current_input.shape}, dtype={current_input.dtype}",
                                        flush=True,
                                    )
                                    # Force clamp BEFORE any other operations
                                    current_input = current_input.clamp(
                                        min=0, max=vocab_size - 1
                                    )
                                    # Raise exception to stop execution and debug
                                    raise RuntimeError(
                                        f"Invalid tokens detected in async loop step {step}: "
                                        f"min={min_token}, max={max_token}, vocab={vocab_size}"
                                    )

                        # Assuming past_kv contains cached sequence length info
                        outputs = self._model(  # type: ignore
                            current_input,
                            past_key_values=current_past_kv,
                            use_cache=True,
                        )
                        # Update past_key_values for next step
                        if outputs.past_key_values is not None:
                            current_past_kv = outputs.past_key_values
                            last_past_kv = current_past_kv
                    else:
                        # CRITICAL: Final validation RIGHT before model forward (no past_kv case)
                        if hasattr(self._model, "config"):
                            vocab_size = getattr(self._model.config, "vocab_size", None)
                            if vocab_size is not None:
                                # Check for invalid tokens using safe operations
                                has_invalid = (current_input >= vocab_size).any() or (
                                    current_input < 0
                                ).any()
                                if has_invalid:
                                    # Calculate invalid count safely
                                    invalid_count = (
                                        (
                                            (current_input >= vocab_size)
                                            | (current_input < 0)
                                        )
                                        .sum()
                                        .item()
                                    )

                                    # Get min/max safely - move to CPU first to avoid CUDA errors
                                    try:
                                        max_token = current_input.cpu().max().item()
                                        min_token = current_input.cpu().min().item()
                                    except Exception as cpu_err:
                                        # If even CPU fails, tensor is severely corrupted
                                        print(
                                            f"[CRITICAL ERROR] Async loop step {step} (no past_kv): Cannot read tensor values! "
                                            f"Error: {cpu_err}",
                                            flush=True,
                                        )
                                        max_token = vocab_size  # Assume worst case
                                        min_token = -1

                                    print(
                                        f"[CRITICAL ERROR] Async loop step {step} (no past_kv): Invalid tokens RIGHT before model forward! "
                                        f"min={min_token}, max={max_token}, vocab={vocab_size}, "
                                        f"invalid={invalid_count}/{current_input.numel()}",
                                        flush=True,
                                    )
                                    # Force clamp BEFORE any other operations
                                    current_input = current_input.clamp(
                                        min=0, max=vocab_size - 1
                                    )
                                    # Raise exception to stop execution and debug
                                    raise RuntimeError(
                                        f"Invalid tokens detected in async loop step {step}: "
                                        f"min={min_token}, max={max_token}, vocab={vocab_size}"
                                    )

                        outputs = self._model(current_input, use_cache=True)  # type: ignore
                        # Initialize past_key_values on first step if KV cache enabled
                        if outputs.past_key_values is not None:
                            current_past_kv = outputs.past_key_values
                            last_past_kv = current_past_kv

                    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                    # Get logits for last position
                    next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                    # --- BEGIN PATCH: Stability fix for softmax ---
                    # Sample or greedy
                    if do_sample:
                        # Promote to FP32 for numerical stability
                        logits = next_token_logits.float()

                        # Clamp to prevent overflow/underflow
                        logits = torch.clamp(logits, -50.0, 50.0)

                        # Apply temperature and compute softmax in FP32
                        probs = torch.softmax(logits / temperature, dim=-1)

                        # Recast to half precision only after computing probs
                        if next_token_logits.dtype == torch.float16:
                            probs = probs.half()

                        # Safety check for NaN/Inf/negative values
                        if (
                            torch.isnan(probs).any()
                            or (probs < 0).any()
                            or torch.isinf(probs).any()
                        ):
                            print(
                                "[ERROR] Invalid probs detected: NaN/Inf or negative values"
                            )
                            torch.cuda.synchronize()
                            raise RuntimeError("Softmax instability on GPU")

                        next_token = torch.multinomial(
                            probs, num_samples=1
                        )  # [batch_size, 1]
                    else:
                        # Greedy sampling - apply temperature if needed
                        if temperature != 1.0:
                            next_token_logits = next_token_logits / temperature
                    next_token = torch.argmax(
                        next_token_logits, dim=-1, keepdim=True
                    )  # [batch_size, 1]
                    # --- END PATCH ---

                    # CRITICAL: Validate generated token indices before storing
                    # This prevents invalid token IDs from causing embedding layer crashes
                    if hasattr(self._model, "config"):
                        vocab_size = getattr(self._model.config, "vocab_size", None)
                        if vocab_size is not None:
                            if (next_token >= vocab_size).any() or (
                                next_token < 0
                            ).any():
                                # Get min/max safely - move to CPU first to avoid CUDA errors
                                try:
                                    max_token = next_token.cpu().max().item()
                                    min_token = next_token.cpu().min().item()
                                except Exception:
                                    max_token = vocab_size  # Assume worst case
                                    min_token = -1
                                self.logger.error(
                                    "[HF-WRAPPER] Invalid token index generated: "
                                    "min=%d, max=%d, vocab_size=%d",
                                    min_token,
                                    max_token,
                                    vocab_size,
                                )
                                # Clamp invalid tokens to valid range
                                next_token = next_token.clamp(min=0, max=vocab_size - 1)
                                print(
                                    f"[WARNING] Clamped generated tokens to valid range [0, {vocab_size - 1}]",
                                    flush=True,
                                )

                    # Store generated token and logits
                    generated_tokens.append(next_token)
                    generated_logits.append(
                        next_token_logits.unsqueeze(1)
                    )  # [batch_size, 1, vocab_size]

                    # Append to input for next step
                    # If using KV cache, only pass new token
                    if current_past_kv is not None:
                        current_input = next_token  # Only new token
                    else:
                        current_input = torch.cat([current_input, next_token], dim=1)

                    # Progress print (reduced frequency to reduce CPU overhead)
                    if step % 16 == 0 or step == max_new_tokens - 1:
                        print(
                            f"[GEN] Step {step+1}/{max_new_tokens} | "
                            f"Current seq len: {current_input.shape[-1]}",
                            flush=True,
                        )

                    # Early stopping on EOS (optional)
                    if (
                        hasattr(self._tokenizer, "eos_token_id")
                        and self._tokenizer.eos_token_id is not None
                    ):
                        if (next_token == self._tokenizer.eos_token_id).all():
                            break

                # Concatenate results
                generated_ids = torch.cat(
                    generated_tokens, dim=1
                )  # [batch_size, max_new_tokens]
                logits = torch.cat(
                    generated_logits, dim=1
                )  # [batch_size, max_new_tokens, vocab_size]

                # Store last KV cache if available for reuse
                if last_past_kv is not None:
                    # Store in internal cache format (would need KVCache wrapper)
                    # For now, store raw past_key_values
                    if hasattr(self, "_last_generated_kv"):
                        # Update or create cache entry
                        # Note: This is simplified - full implementation would use KVCache wrapper
                        self._last_generated_kv_raw = last_past_kv

                return generated_ids, logits

    def _generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens using cached KV states.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_ids, logits)
        """
        # Use forward() directly to avoid generate() cache_position issues
        # Only pass the new tokens (not cached ones) as input
        cache_len = self._kv_cache.seq_len if self._kv_cache else 0
        new_input_ids = input_ids[:, cache_len:]

        if new_input_ids.shape[1] == 0:
            # No new tokens to process - shouldn't happen in normal flow
            raise ValueError(
                f"No new tokens after cache (cache_len={cache_len}, "
                f"input_len={input_ids.shape[1]})"
            )

        # CRITICAL: Validate new_input_ids before forward pass
        if hasattr(self._model, "config"):
            vocab_size = getattr(self._model.config, "vocab_size", None)
            if vocab_size is not None:
                if (new_input_ids >= vocab_size).any() or (new_input_ids < 0).any():
                    # Get min/max safely - move to CPU first to avoid CUDA errors
                    try:
                        max_token = new_input_ids.cpu().max().item()
                        min_token = new_input_ids.cpu().min().item()
                    except Exception:
                        max_token = vocab_size  # Assume worst case
                        min_token = -1
                    print(
                        f"[ERROR] Invalid tokens in _generate_with_kv_cache: "
                        f"min={min_token}, max={max_token}, vocab={vocab_size}",
                        flush=True,
                    )
                    # Clamp invalid tokens
                    new_input_ids = new_input_ids.clamp(min=0, max=vocab_size - 1)

        # Run forward pass with cached KV
        with torch.no_grad():
            outputs = self._model(  # type: ignore
                new_input_ids,
                past_key_values=(
                    self._kv_cache.past_key_values if self._kv_cache else None
                ),
                use_cache=True,
            )

        logits = outputs.logits  # [batch, seq, vocab]
        past_key_values = outputs.past_key_values

        # Sample from logits for max_new_tokens
        generated_tokens = []
        all_logits = []

        # First token from current logits
        next_token_logits = logits[:, -1, :] / temperature
        all_logits.append(next_token_logits)

        if do_sample:
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        # CRITICAL: Validate generated token indices
        if hasattr(self._model, "config"):
            vocab_size = getattr(self._model.config, "vocab_size", None)
            if vocab_size is not None:
                if (next_token >= vocab_size).any() or (next_token < 0).any():
                    # Get min/max safely - move to CPU first to avoid CUDA errors
                    try:
                        max_token = next_token.cpu().max().item()
                        min_token = next_token.cpu().min().item()
                    except Exception:
                        max_token = vocab_size  # Assume worst case
                        min_token = -1
                    # Clamp invalid tokens to valid range
                    next_token = next_token.clamp(min=0, max=vocab_size - 1)
                    print(
                        f"[WARNING] Clamped first generated token to valid range [0, {vocab_size - 1}] "
                        f"(min={min_token}, max={max_token})",
                        flush=True,
                    )

        generated_tokens.append(next_token)

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            # CRITICAL: Validate next_token before each forward pass
            if hasattr(self._model, "config"):
                vocab_size = getattr(self._model.config, "vocab_size", None)
                if vocab_size is not None:
                    if (next_token >= vocab_size).any() or (next_token < 0).any():
                        # Get min/max safely - move to CPU first to avoid CUDA errors
                        try:
                            max_token = next_token.cpu().max().item()
                            min_token = next_token.cpu().min().item()
                        except Exception:
                            max_token = vocab_size  # Assume worst case
                            min_token = -1
                        print(
                            f"[ERROR] Invalid next_token in _generate_with_kv_cache loop: "
                            f"min={min_token}, max={max_token}, vocab={vocab_size}",
                            flush=True,
                        )
                        # Clamp invalid tokens
                        next_token = next_token.clamp(min=0, max=vocab_size - 1)

            with torch.no_grad():
                outputs = self._model(  # type: ignore
                    next_token, past_key_values=past_key_values, use_cache=True
                )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :] / temperature
            all_logits.append(next_token_logits)

            if do_sample:
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # CRITICAL: Validate generated token indices
            if hasattr(self._model, "config"):
                vocab_size = getattr(self._model.config, "vocab_size", None)
                if vocab_size is not None:
                    if (next_token >= vocab_size).any() or (next_token < 0).any():
                        # Get min/max safely - move to CPU first to avoid CUDA errors
                        try:
                            max_token = next_token.cpu().max().item()
                            min_token = next_token.cpu().min().item()
                        except Exception:
                            max_token = vocab_size  # Assume worst case
                            min_token = -1
                        # Clamp invalid tokens to valid range
                        next_token = next_token.clamp(min=0, max=vocab_size - 1)
                        print(
                            f"[WARNING] Clamped generated token to valid range [0, {vocab_size - 1}] "
                            f"(min={min_token}, max={max_token})",
                            flush=True,
                        )

            generated_tokens.append(next_token)

        # Concatenate results
        generated_ids = torch.cat(generated_tokens, dim=1)
        logits_stacked = torch.stack(all_logits, dim=1)

        # Store updated KV cache
        if past_key_values is not None:
            self._last_generated_kv = KVCache.from_hf_output(past_key_values)
        else:
            self._last_generated_kv = None

        return generated_ids, logits_stacked

    def get_last_generated_kv(self) -> Optional[KVCache]:
        """Get the KV cache from the last generation."""
        return getattr(self, "_last_generated_kv", None)

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer information for compatibility checking."""
        return {
            "model_name": self._model_name,
            "vocab_size": self._tokenizer.vocab_size,  # type: ignore
            "pad_token_id": self._tokenizer.pad_token_id,  # type: ignore
            "eos_token_id": self._tokenizer.eos_token_id,  # type: ignore
            "bos_token_id": self._tokenizer.bos_token_id,  # type: ignore
            "unk_token_id": self._tokenizer.unk_token_id,  # type: ignore
        }

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, return_tensors="pt")  # type: ignore

    def encode_with_attention_mask(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs with attention mask."""
        inputs = self._tokenizer(  # type: ignore
            text, return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def decode(self, token_ids) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, list):
            # Convert list to tensor and flatten if needed
            token_ids = torch.tensor(token_ids)
        elif not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)

        # Ensure we have a 1D tensor for decoding
        if token_ids.dim() > 1:
            token_ids = token_ids.flatten()

        return self._tokenizer.decode(  # type: ignore[union-attr]
            token_ids.tolist(), skip_special_tokens=True
        )

    @property
    def device(self) -> str:
        """Get the device this model is running on."""
        return self._device

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def model(self):
        """Get the underlying model (read-only)."""
        return self._model

    def optimize(self, optimization_manager) -> None:
        """Optimize the model using the provided optimization manager."""
        if optimization_manager and hasattr(self, "_model") and self._model is not None:
            self._model = optimization_manager.optimize_model(self._model)

    def supports_kv_append(self) -> bool:
        """
        Check if this model supports KV cache appending.

        Returns:
            True if KV append is enabled and model is a causal LM
        """
        if not self._kv_append_enabled:
            return False

        # HF causal LMs support past_key_values
        return hasattr(self._model, "forward") and hasattr(self._model, "config")

    def get_kv_cache(self) -> Optional[KVCache]:
        """
        Get the current KV cache from the model.

        Returns:
            KVCache object or None if not available
        """
        return self._kv_cache

    def append_kv_cache(self, kv_chunk: KVCache) -> None:
        """
        Append a KV cache chunk to the model's cache.

        Args:
            kv_chunk: KVCache object containing keys and values to append
        """
        if not self.supports_kv_append():
            self.logger.warning("KV append not supported, ignoring")
            return

        # Import kernel here to avoid circular dependency
        try:
            from kernels import get_kv_append

            kv_append_fn = get_kv_append(self._device)
        except ImportError:
            kv_append_fn = None

        if self._kv_cache is None:
            # First chunk, just store it
            self._kv_cache = kv_chunk.to(torch.device(self._device))
            self.logger.debug(f"Initialized KV cache with {kv_chunk.seq_len} positions")
        else:
            # Append to existing cache
            if kv_append_fn is not None:
                # Use kernel for efficient append
                self._kv_cache = self._append_kv_with_kernel(
                    kv_append_fn, self._kv_cache, kv_chunk
                )
            else:
                # Fallback to PyTorch concat
                self._kv_cache = self._append_kv_pytorch(self._kv_cache, kv_chunk)

            self.logger.debug(
                f"Appended {kv_chunk.seq_len} positions to KV cache, "
                f"total: {self._kv_cache.seq_len}"
            )

    def _append_kv_pytorch(self, base_cache: KVCache, new_cache: KVCache) -> KVCache:
        """
        Append KV cache using PyTorch concat (fallback).

        Args:
            base_cache: Existing base cache
            new_cache: New cache to append

        Returns:
            Updated KVCache
        """
        # Ensure same device
        new_cache = new_cache.to(torch.device(self._device))

        # Concatenate along sequence dimension (dim=2)
        appended_kv = tuple(
            (
                torch.cat([base_k, new_k], dim=2),
                torch.cat([base_v, new_v], dim=2),
            )
            for (base_k, base_v), (new_k, new_v) in zip(
                base_cache.past_key_values, new_cache.past_key_values
            )
        )

        return KVCache(
            past_key_values=appended_kv,
            seq_len=base_cache.seq_len + new_cache.seq_len,
            dtype=base_cache.dtype,
            device=base_cache.device,
        )

    def _append_kv_with_kernel(
        self, kv_append_fn: Any, base_cache: KVCache, new_cache: KVCache
    ) -> KVCache:
        """
        Append KV cache using kernel implementation.

        Args:
            kv_append_fn: Kernel function for KV append
            base_cache: Existing base cache
            new_cache: New cache to append

        Returns:
            Updated KVCache
        """
        # Ensure same device
        new_cache = new_cache.to(torch.device(self._device))

        # Use kernel for each layer
        appended_kv = []
        for (base_k, base_v), (new_k, new_v) in zip(
            base_cache.past_key_values, new_cache.past_key_values
        ):
            try:
                # Call kernel (signature: base_k, base_v, new_k, new_v)
                out_k, out_v = kv_append_fn(base_k, base_v, new_k, new_v)
                appended_kv.append((out_k, out_v))
            except Exception as e:
                self.logger.warning(
                    f"Kernel append failed, falling back to PyTorch: {e}"
                )
                # Fallback to concat
                out_k = torch.cat([base_k, new_k], dim=2)
                out_v = torch.cat([base_v, new_v], dim=2)
                appended_kv.append((out_k, out_v))

        # Synchronize CUDA stream if on CUDA device
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        return KVCache(
            past_key_values=tuple(appended_kv),
            seq_len=base_cache.seq_len + new_cache.seq_len,
            dtype=base_cache.dtype,
            device=base_cache.device,
        )

    def clear_kv_cache(self) -> None:
        """Clear the model's KV cache."""
        self._kv_cache = None
        self.logger.debug("Cleared KV cache")

    def cleanup(self) -> None:
        """Clean up model memory (useful for MPS)."""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            del self._tokenizer

        # Force garbage collection
        import gc

        gc.collect()

        # Clear MPS cache if using MPS
        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.logger.info("Model memory cleaned up")


def create_tiny_hf_wrapper(
    model_name: str = "sshleifer/tiny-gpt2",
    device: str = "auto",
    max_memory_mb: int = 500,
    tokenizer: Optional[AutoTokenizer] = None,
) -> HFWrapper:
    """
    Create a Hugging Face wrapper with tiny model defaults for testing.

    Args:
        model_name: Tiny model name (default: sshleifer/tiny-gpt2)
        device: Device to run on
        max_memory_mb: Maximum memory usage in MB
        tokenizer: Shared tokenizer instance

    Returns:
        HFWrapper instance configured for tiny models
    """
    return HFWrapper(
        model_name=model_name,
        device=device,
        torch_dtype=None,  # Auto-select based on device
        max_memory_mb=max_memory_mb,
        tokenizer=tokenizer,
    )
