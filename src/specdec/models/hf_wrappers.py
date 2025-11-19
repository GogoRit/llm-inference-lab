"""
Hugging Face Model Wrappers for Speculative Decoding

Provides HFWrapper that wraps Hugging Face models with safe defaults
for tiny models only (smoke runs) and MPS memory hygiene.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..cache.kv_types import KVCache
from ..utils.interfaces import LanguageModel
from ..utils.token_validation import get_vocab_size, validate_and_clamp_tokens


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
        self._last_generated_kv: Optional[KVCache] = None  # Type annotation for mypy
        self._kv_append_enabled = os.getenv(
            "SPECDEC_ENABLE_KV_APPEND", "1"
        ).lower() in (
            "1",
            "true",
            "yes",
        )

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

            # Load model with memory considerations and SDPA enabled
            model_kwargs = {
                "torch_dtype": self._torch_dtype,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",  # Enable PyTorch 2.0 SDPA
            }

            # Ensure SDPA is available (PyTorch 2.0+)
            try:
                import torch.nn.functional as F

                if not hasattr(F, "scaled_dot_product_attention"):
                    self.logger.warning(
                        "PyTorch 2.0+ required for SDPA. Falling back to default attention."
                    )
                    # Remove attn_implementation if SDPA not available
                    model_kwargs.pop("attn_implementation", None)
            except ImportError:
                model_kwargs.pop("attn_implementation", None)

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
        attention_mask: Optional[torch.Tensor] = None,
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
            # Use centralized validation helper
            vocab_size = get_vocab_size(self)
            if vocab_size is not None:
                input_ids = validate_and_clamp_tokens(
                    input_ids, vocab_size, "generate_tokens"
                )

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

                # Re-validate after device transfer (trust device transfer, but safety check)
                vocab_size = get_vocab_size(self)
                if vocab_size is not None:
                    input_ids = validate_and_clamp_tokens(
                        input_ids, vocab_size, "generate_tokens_after_transfer"
                    )

                # Use KV-aware generation if supported and enabled
                # Check if past_key_values are provided (from pre-allocated cache manager)
                past_key_values = kwargs.pop("past_key_values", None)
                current_seq_lens = kwargs.pop("current_seq_lens", None)

                if self.supports_kv_append() and (
                    self._kv_cache is not None or past_key_values is not None
                ):
                    return self._generate_with_kv_cache(
                        input_ids,
                        max_new_tokens,
                        temperature,
                        do_sample,
                        past_key_values=past_key_values,
                        current_seq_lens=current_seq_lens,
                        **kwargs,
                    )

                # Standard generation without KV cache
                # Input already validated at entry point - trust validation

                # Use provided attention mask or create default (all ones)
                # This allows batch processing to pass proper masks that ignore padding
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)

                # Extract position_ids from kwargs if provided
                position_ids = kwargs.pop("position_ids", None)

                # Generate tokens
                generate_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "pad_token_id": self._tokenizer.eos_token_id,  # type: ignore
                    "eos_token_id": self._tokenizer.eos_token_id,  # type: ignore
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }

                # Add position_ids if provided (some models support explicit position IDs)
                if position_ids is not None:
                    generate_kwargs["position_ids"] = position_ids

                # Add any remaining kwargs
                generate_kwargs.update(kwargs)

                outputs = self._model.generate(**generate_kwargs)  # type: ignore

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
        # Validate input_ids BEFORE async execution using centralized helper
        vocab_size = get_vocab_size(self)
        if vocab_size is not None:
            input_ids = validate_and_clamp_tokens(
                input_ids, vocab_size, "_generate_tokens_async"
            )

        with torch.cuda.stream(stream):
            with torch.no_grad():
                # Move input to device if needed
                if input_ids.device != torch.device(self._device):
                    input_ids = input_ids.to(self._device)

                # Re-validate after device transfer (trust device transfer, safety check only)
                vocab_size = get_vocab_size(self)
                if vocab_size is not None:
                    input_ids = validate_and_clamp_tokens(
                        input_ids, vocab_size, "async_after_transfer"
                    )

                batch_size, seq_len = input_ids.shape
                generated_tokens = []
                generated_logits = []

                current_input = input_ids
                current_past_kv = past_key_values
                last_past_kv = None  # Track KV cache for return

                for step in range(max_new_tokens):
                    # CRITICAL: Validate current_input before EACH forward pass
                    # Note: We rely on CUDA events for synchronization, not explicit sync
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
                                        f"[CRITICAL ERROR] Async loop step {step}: "
                                        f"Invalid tokens RIGHT before model forward! "
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

                        # Construct attention mask and position IDs for pre-allocated cache
                        # Extract current_seq_lens from kwargs if provided
                        step_current_seq_lens = kwargs.get("current_seq_lens", None)

                        # Construct attention mask for pre-allocated buffer
                        # CRITICAL: Pass input_ids to calculate correct target_length (cache_len + input_len)
                        step_attention_mask = (
                            self._construct_attention_mask_for_preallocated_cache(
                                past_key_values=current_past_kv,
                                current_seq_lens=step_current_seq_lens,
                                batch_size=current_input.shape[0],
                                device=current_input.device,
                                input_ids=current_input,
                            )
                        )

                        # Construct position IDs
                        if step_current_seq_lens is not None:
                            step_position_ids = (
                                self._construct_position_ids_for_preallocated_cache(
                                    current_seq_lens=step_current_seq_lens,
                                    batch_size=current_input.shape[0],
                                    device=current_input.device,
                                )
                            )
                        else:
                            step_position_ids = None

                        # Prepare forward pass arguments
                        forward_kwargs = {
                            "input_ids": current_input,
                            "past_key_values": current_past_kv,
                            "use_cache": True,
                        }

                        if step_attention_mask is not None:
                            forward_kwargs["attention_mask"] = step_attention_mask
                        if step_position_ids is not None:
                            forward_kwargs["position_ids"] = step_position_ids

                        # Assuming past_kv contains cached sequence length info
                        outputs = self._model(**forward_kwargs)  # type: ignore
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
                                            f"[CRITICAL ERROR] Async loop step {step} "
                                            f"(no past_kv): Cannot read tensor values! "
                                            f"Error: {cpu_err}",
                                            flush=True,
                                        )
                                        max_token = vocab_size  # Assume worst case
                                        min_token = -1

                                    print(
                                        f"[CRITICAL ERROR] Async loop step {step} "
                                        f"(no past_kv): Invalid tokens RIGHT before model forward! "
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
                            self.logger.error(
                                "[ERROR] Invalid probs detected: NaN/Inf or negative values"
                            )
                            # Use event-based sync if available, otherwise fallback to sync
                            if hasattr(torch.cuda, "current_stream"):
                                torch.cuda.current_stream().synchronize()
                            raise RuntimeError("Softmax instability on GPU")

                        # CRITICAL: Ensure probs are valid before multinomial
                        if (
                            torch.isnan(probs).any()
                            or torch.isinf(probs).any()
                            or (probs < 0).any()
                        ):
                            print(
                                "[ERROR] Invalid probabilities before multinomial in async!",
                                flush=True,
                            )
                            # Fallback to argmax if probs are invalid
                            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                        else:
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

                    # CRITICAL: Ensure dtype is long/int64 and validate IMMEDIATELY
                    next_token = next_token.long()

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
                                print(
                                    f"[CRITICAL ERROR] Invalid token from argmax/multinomial in async! "
                                    f"min={min_token}, max={max_token}, vocab={vocab_size}",
                                    flush=True,
                                )
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

    def _construct_attention_mask_for_preallocated_cache(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        current_seq_lens: Optional[List[int]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        input_ids: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Construct attention mask for pre-allocated KV cache buffers.

        When using pre-allocated buffers, the past_key_values may be views from
        a larger buffer (Max_L) but only valid up to current_seq_len. This method
        creates an attention mask that masks out invalid positions.

        CRITICAL: The mask length MUST equal past_key_values.shape[2] + input_ids.shape[1]
        to match the actual sequence length being processed by the model.

        Args:
            past_key_values: KV cache tuple from cache manager (may be views)
            current_seq_lens: List of current sequence lengths per batch entry (cache lengths)
            batch_size: Batch size
            device: Device for mask tensor
            input_ids: Input token IDs [batch_size, seq_len] (new tokens being added)
            target_length: Target sequence length for mask (cache_len + input_len).
                          If None, will be calculated from past_key_values and input_ids.

        Returns:
            attention_mask: [batch_size, 1, 1, target_length] mask (1 for valid, 0 for invalid)
                           or None if not needed
        """
        if past_key_values is None or len(past_key_values) == 0:
            return None

        # Infer cache length from first layer's key tensor
        first_key = past_key_values[0][0]
        cache_len = first_key.shape[2]  # [B, H, seq_len, D] -> seq_len dimension

        if device is None:
            device = first_key.device

        # Calculate target_length: cache_len + input_ids.shape[1]
        if target_length is None:
            if input_ids is not None:
                input_len = input_ids.shape[1]
                target_length = cache_len + input_len
            else:
                # Fallback: use cache_len if no input_ids provided
                # This is for backward compatibility but may cause issues
                target_length = cache_len
        else:
            # Use provided target_length
            pass

        # If current_seq_lens not provided, infer from past_key_values shape
        if current_seq_lens is None:
            # Assume all sequences have same cache length
            current_seq_lens = [cache_len] * batch_size
        elif len(current_seq_lens) != batch_size:
            # Pad or truncate to match batch_size
            if len(current_seq_lens) < batch_size:
                current_seq_lens = current_seq_lens + [current_seq_lens[-1]] * (
                    batch_size - len(current_seq_lens)
                )
            else:
                current_seq_lens = current_seq_lens[:batch_size]

        # Calculate total sequence length for each batch entry: current_seq_lens[b] + input_len
        # CRITICAL: Use current_seq_lens[b] (actual valid cache length) not cache_len (buffer size)
        if input_ids is not None:
            input_len = input_ids.shape[1]
            total_seq_lens = [
                current_seq_lens[b] + input_len for b in range(batch_size)
            ]
        else:
            # Fallback: use current_seq_lens as total (assumes no new input)
            total_seq_lens = current_seq_lens

        # CRITICAL: Get model dtype for SDPA compatibility
        # SDPA expects float masks with additive semantics (0.0 = keep, -inf = mask)
        model_dtype = self._torch_dtype
        if hasattr(self, "_model") and hasattr(self._model, "dtype"):
            model_dtype = self._model.dtype
        elif hasattr(self, "_model") and hasattr(self._model, "config"):
            # Fallback: try to get dtype from config
            if hasattr(self._model.config, "torch_dtype"):
                model_dtype = self._model.config.torch_dtype

        # Construct attention mask: [batch_size, 1, 1, target_length]
        # CRITICAL: Use additive masking for SDPA (0.0 = attend, -inf = mask)
        # Initialize with -inf (masked) for all positions
        min_value = torch.finfo(model_dtype).min
        attention_mask = torch.full(
            (batch_size, 1, 1, target_length),
            min_value,
            dtype=model_dtype,
            device=device,
        )

        # Set mask to 0.0 for valid positions (0 to total_seq_len-1)
        # SDPA treats 0.0 as "attend" and -inf as "mask"
        # total_seq_len = current_seq_lens[b] + input_len (actual valid sequence length)
        for b, total_len in enumerate(total_seq_lens):
            # Clamp total_len to target_length (safety check)
            valid_len = min(total_len, target_length)
            if valid_len > 0:
                attention_mask[b, 0, 0, :valid_len] = 0.0

        return attention_mask

    def _construct_position_ids_for_preallocated_cache(
        self,
        current_seq_lens: List[int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct position IDs for pre-allocated KV cache.

        Position IDs should start from current_seq_len for each batch entry,
        since we're decoding the next token.

        Args:
            current_seq_lens: List of current sequence lengths per batch entry
            batch_size: Batch size
            device: Device for position IDs tensor

        Returns:
            position_ids: [batch_size, 1] tensor with position = current_seq_len
        """
        if len(current_seq_lens) != batch_size:
            # Pad or truncate
            if len(current_seq_lens) < batch_size:
                current_seq_lens = current_seq_lens + [current_seq_lens[-1]] * (
                    batch_size - len(current_seq_lens)
                )
            else:
                current_seq_lens = current_seq_lens[:batch_size]

        # Position IDs for decoding: position = current_seq_len (next token position)
        position_ids = torch.tensor(
            current_seq_lens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(
            1
        )  # [batch_size, 1]

        return position_ids

    def _generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        current_seq_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens using cached KV states with support for pre-allocated buffers.

        Args:
            input_ids: Input token IDs [batch_size, seq_len] (new tokens only)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            past_key_values: Optional KV cache from pre-allocated buffer manager
            attention_mask: Optional attention mask (constructed if not provided)
            position_ids: Optional position IDs (constructed if not provided)
            current_seq_lens: Optional list of current sequence lengths per batch entry
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_ids, logits)
        """
        # Use past_key_values from parameter if provided, otherwise fall back to internal cache
        if past_key_values is None:
            past_key_values = self._kv_cache.past_key_values if self._kv_cache else None

        # Determine cache length from past_key_values or internal cache
        if past_key_values is not None and len(past_key_values) > 0:
            # Infer from first layer's key tensor shape
            first_key = past_key_values[0][0]
            cache_len = first_key.shape[2]  # [B, H, seq_len, D]
        else:
            cache_len = self._kv_cache.seq_len if self._kv_cache else 0

        # Extract new tokens (skip cached portion)
        new_input_ids = input_ids[:, cache_len:] if cache_len > 0 else input_ids

        if new_input_ids.shape[1] == 0:
            # No new tokens to process - shouldn't happen in normal flow
            raise ValueError(
                f"No new tokens after cache (cache_len={cache_len}, "
                f"input_len={input_ids.shape[1]})"
            )

        batch_size = new_input_ids.shape[0]
        device = new_input_ids.device

        # CRITICAL: Construct attention mask for pre-allocated buffers
        # This masks out invalid positions in the pre-allocated buffer
        # CRITICAL: Pass input_ids to calculate correct target_length (cache_len + input_len)
        if attention_mask is None and past_key_values is not None:
            attention_mask = self._construct_attention_mask_for_preallocated_cache(
                past_key_values=past_key_values,
                current_seq_lens=current_seq_lens,
                batch_size=batch_size,
                device=device,
                input_ids=new_input_ids,
            )

        # CRITICAL: Construct position IDs based on current sequence lengths
        if position_ids is None and current_seq_lens is not None:
            position_ids = self._construct_position_ids_for_preallocated_cache(
                current_seq_lens=current_seq_lens,
                batch_size=batch_size,
                device=device,
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

        # Prepare forward pass arguments
        forward_kwargs = {
            "input_ids": new_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

        # Add attention_mask if constructed or provided
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask

        # Add position_ids if constructed or provided
        if position_ids is not None:
            forward_kwargs["position_ids"] = position_ids

        # Run forward pass with cached KV and proper masking
        with torch.no_grad():
            outputs = self._model(**forward_kwargs)  # type: ignore

        logits = outputs.logits  # [batch, seq, vocab]
        past_key_values = outputs.past_key_values

        # Track current sequence lengths for attention mask construction
        # Initialize from cache_len or current_seq_lens
        if current_seq_lens is None:
            # Infer from past_key_values if available
            if past_key_values is not None and len(past_key_values) > 0:
                first_key = past_key_values[0][0]
                inferred_len = first_key.shape[2]  # [B, H, seq_len, D]
                step_current_seq_lens = [inferred_len] * batch_size
            else:
                step_current_seq_lens = [cache_len] * batch_size
        else:
            step_current_seq_lens = list(current_seq_lens)  # Copy for mutation

        # Sample from logits for max_new_tokens
        generated_tokens = []
        all_logits = []

        # First token from current logits
        next_token_logits = logits[:, -1, :] / temperature
        all_logits.append(next_token_logits)

        if do_sample:
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # CRITICAL: Ensure probs are valid before multinomial
            if (
                torch.isnan(probs).any()
                or torch.isinf(probs).any()
                or (probs < 0).any()
            ):
                print("[ERROR] Invalid probabilities before multinomial!", flush=True)
                # Fallback to argmax if probs are invalid
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        # CRITICAL: Validate generated token indices IMMEDIATELY
        # Ensure dtype is long/int64 for proper token ID representation
        next_token = next_token.long()

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
                        f"[CRITICAL ERROR] Invalid token from argmax/multinomial! "
                        f"min={min_token}, max={max_token}, vocab={vocab_size}",
                        flush=True,
                    )
                    # Clamp invalid tokens to valid range
                    next_token = next_token.clamp(min=0, max=vocab_size - 1)
                    print(
                        f"[WARNING] Clamped first generated token to valid range [0, {vocab_size - 1}] "
                        f"(min={min_token}, max={max_token})",
                        flush=True,
                    )

        generated_tokens.append(next_token)

        # Update sequence lengths after first token
        step_current_seq_lens = [len + 1 for len in step_current_seq_lens]

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

            # Construct attention mask and position IDs for this decoding step
            # past_key_values shape: [B, H, current_seq_len, D] (view from pre-allocated buffer)
            if past_key_values is not None and len(past_key_values) > 0:
                # Construct attention mask for updated sequence length
                # CRITICAL: Pass input_ids (next_token) to calculate correct target_length (cache_len + 1)
                step_attention_mask = (
                    self._construct_attention_mask_for_preallocated_cache(
                        past_key_values=past_key_values,
                        current_seq_lens=step_current_seq_lens,
                        batch_size=batch_size,
                        device=next_token.device,
                        input_ids=next_token,
                    )
                )

                # Construct position IDs (current position = current_seq_len)
                step_position_ids = self._construct_position_ids_for_preallocated_cache(
                    current_seq_lens=step_current_seq_lens,
                    batch_size=batch_size,
                    device=next_token.device,
                )
            else:
                step_attention_mask = None
                step_position_ids = None

            with torch.no_grad():
                forward_kwargs = {
                    "input_ids": next_token,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }

                if step_attention_mask is not None:
                    forward_kwargs["attention_mask"] = step_attention_mask
                if step_position_ids is not None:
                    forward_kwargs["position_ids"] = step_position_ids

                outputs = self._model(**forward_kwargs)  # type: ignore

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Update sequence lengths after this token
            step_current_seq_lens = [len + 1 for len in step_current_seq_lens]

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

    def decode(self, token_ids: Any) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode (torch.Tensor, list, or other iterable)

        Returns:
            Decoded text string
        """
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
