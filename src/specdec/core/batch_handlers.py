"""
Batch Processing Handlers for Speculative Decoding

This module contains specialized handler classes that encapsulate specific
responsibilities in the batch speculative decoding pipeline. This separation
improves maintainability, testability, and readability.

Each handler focuses on a single responsibility:
- DraftGenerationHandler: Draft token generation
- VerificationHandler: Base model verification
- AcceptanceHandler: Token acceptance logic
- RollbackHandler: Zero-copy pointer rollback
- KVCacheHandler: KV cache management
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..utils.token_validation import get_vocab_size, validate_and_clamp_tokens
from .sequence_utils import create_position_ids

logger = logging.getLogger(__name__)


class DraftGenerationHandler:
    """Handles draft token generation in batch mode."""

    def __init__(
        self,
        draft_lm: Any,
        device: str,
        kv_cache_manager: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the draft generation handler.

        Args:
            draft_lm: Draft language model
            device: Device to run on
            kv_cache_manager: KV cache manager instance
            logger: Optional logger instance
        """
        self.draft_lm = draft_lm
        self.device = device
        self.kv_cache_manager = kv_cache_manager
        self.logger = logger or logging.getLogger(__name__)

    def generate_draft_tokens(
        self,
        active_input_ids: torch.Tensor,
        active_attention_mask: Optional[torch.Tensor],
        active_position_ids: Optional[torch.Tensor],
        active_indices: torch.Tensor,
        relative_indices_tensor: torch.Tensor,
        current_seq_lens: List[int],
        k: int,
        temperature: float,
        do_sample: bool,
        draft_stream: Optional[torch.cuda.Stream] = None,
        kv_cache_enabled: bool = True,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        Optional[torch.cuda.Event],
        Optional[torch.cuda.Event],
    ]:
        """
        Generate draft tokens for active sequences in batch.

        Args:
            active_input_ids: Input token IDs for active sequences [active_count, seq_len]
            active_attention_mask: Attention mask for active sequences
            active_position_ids: Position IDs for active sequences
            active_indices: Indices of active sequences in batch
            relative_indices_tensor: Relative indices for KV cache (0 to active_count-1)
            current_seq_lens: Current sequence lengths per batch position
            k: Number of draft tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            draft_stream: Optional CUDA stream for async execution
            kv_cache_enabled: Whether KV cache is enabled
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (draft_tokens, draft_logits, draft_time_ms, draft_start_event, draft_end_event)
        """
        # Setup CUDA events for timing if using streams
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

        draft_start_wall = time.time()

        # Apply temperature stabilization for draft model
        # CRITICAL: Use same temperature and sampling strategy as base for better acceptance
        # When both use greedy (do_sample=False), they should match exactly
        # When both use sampling, use same temperature for consistency
        if not do_sample:
            # Greedy mode: use same temperature (typically 1.0 or very low)
            draft_temperature = temperature if temperature is not None else 1.0
            draft_do_sample = False  # Match base model's greedy mode
        else:
            # Sampling mode: use same temperature for better alignment
            # Lower temperature can help, but exact match is better for acceptance
            draft_temperature = temperature if temperature is not None else 0.7
            draft_do_sample = do_sample  # Match base model's sampling mode

        # Prepare past_key_values for draft model
        draft_past_kv = None
        draft_current_seq_lens_for_model = None
        if kv_cache_enabled:
            draft_past_kv = self.kv_cache_manager.get_draft_past_kv(
                relative_indices_tensor
            )
            draft_current_seq_lens_for_model = [
                (
                    self.kv_cache_manager.draft_current_seq_lens[i]
                    if i < len(self.kv_cache_manager.draft_current_seq_lens)
                    else current_seq_lens[active_indices[i]]
                )
                for i in range(len(active_indices))
            ]

        # Prepare input tensor for streams (clone once for independence)
        draft_vocab_size = get_vocab_size(self.draft_lm)
        active_input_ids_prepared = None

        if draft_stream is not None or kwargs.get("verify_stream") is not None:
            try:
                active_input_ids_prepared = (
                    active_input_ids.detach().clone().contiguous()
                )
                if draft_vocab_size is not None:
                    active_input_ids_prepared = validate_and_clamp_tokens(
                        active_input_ids_prepared,
                        draft_vocab_size,
                        "batch_input_pre_stream",
                        strict=True,
                    )
            except Exception as e:
                self.logger.error(f"Failed to prepare tensor for streams: {e}")
                raise RuntimeError(
                    f"Failed to prepare tensor for CUDA streams: {e}"
                ) from e

        # Generate draft tokens
        if draft_stream is not None:
            with torch.cuda.stream(draft_stream):
                try:
                    draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                        active_input_ids_prepared,
                        max_new_tokens=k,
                        temperature=draft_temperature,
                        do_sample=draft_do_sample,  # Match base model's sampling strategy
                        stream=draft_stream,
                        past_key_values=draft_past_kv,
                        attention_mask=active_attention_mask,
                        position_ids=active_position_ids,
                        **kwargs,
                    )
                except RuntimeError as e:
                    if "indexSelectLargeIndex" in str(e) or "device-side assert" in str(
                        e
                    ):
                        vocab_info = (
                            getattr(
                                self.draft_lm._model.config, "vocab_size", "unknown"
                            )
                            if hasattr(self.draft_lm, "_model")
                            else "unknown"
                        )
                        self.logger.error(
                            f"CUDA embedding error in draft model! "
                            f"active_input_ids.shape={active_input_ids.shape}, "
                            f"vocab_size={vocab_info}, Error: {e}"
                        )
                    raise
            if draft_end_event is not None:
                draft_end_event.record(draft_stream)
        else:
            # Non-stream path
            if draft_vocab_size is not None:
                active_input_ids = validate_and_clamp_tokens(
                    active_input_ids,
                    draft_vocab_size,
                    "draft_input_sync",
                    strict=True,
                )

            draft_tokens, draft_logits = self.draft_lm.generate_tokens(
                active_input_ids,
                max_new_tokens=k,
                temperature=draft_temperature,
                do_sample=draft_do_sample,  # Match base model's sampling strategy
                past_key_values=draft_past_kv,
                attention_mask=active_attention_mask,
                position_ids=active_position_ids,
                current_seq_lens=draft_current_seq_lens_for_model,
                **kwargs,
            )

        # Update draft KV cache
        if kv_cache_enabled and len(active_indices) > 0:
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
                    draft_new_kv = self.draft_lm._last_generated_kv.past_key_values
                else:
                    draft_new_kv = self.draft_lm._last_generated_kv

            if draft_new_kv is not None:
                if isinstance(draft_new_kv, tuple):
                    self.kv_cache_manager.update_draft_cache(draft_new_kv)
                elif len(draft_new_kv) > 0:
                    self.kv_cache_manager.update_draft_cache(tuple(draft_new_kv))

        # Validate draft outputs
        if draft_tokens.numel() == 0 or draft_tokens.shape[1] == 0:
            self.logger.error(
                f"Draft model returned empty tokens! Shape: {draft_tokens.shape}"
            )
            raise RuntimeError("Draft model returned empty tokens")

        # Calculate draft time
        if draft_start_event is not None and draft_end_event is not None:
            draft_end_event.synchronize()
            draft_time_ms = draft_start_event.elapsed_time(draft_end_event)
        else:
            draft_time_ms = (time.time() - draft_start_wall) * 1000

        return (
            draft_tokens,
            draft_logits,
            draft_time_ms,
            draft_start_event,
            draft_end_event,
        )


class VerificationHandler:
    """Handles base model verification of draft tokens."""

    def __init__(
        self,
        base_lm: Any,
        device: str,
        kv_cache_manager: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the verification handler.

        Args:
            base_lm: Base language model
            device: Device to run on
            kv_cache_manager: KV cache manager instance
            logger: Optional logger instance
        """
        self.base_lm = base_lm
        self.device = device
        self.kv_cache_manager = kv_cache_manager
        self.logger = logger or logging.getLogger(__name__)

    def verify_draft_tokens(
        self,
        active_input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        active_attention_mask: Optional[torch.Tensor],
        active_position_ids: Optional[torch.Tensor],
        active_indices: torch.Tensor,
        relative_indices_tensor: torch.Tensor,
        current_seq_lens: List[int],
        temperature: float,
        do_sample: bool,
        verify_stream: Optional[torch.cuda.Stream] = None,
        draft_end_event: Optional[torch.cuda.Event] = None,
        kv_cache_enabled: bool = True,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        Optional[torch.cuda.Event],
        Optional[torch.cuda.Event],
    ]:
        """
        Verify draft tokens with base model.

        Args:
            active_input_ids: Input token IDs for active sequences
            draft_tokens: Draft tokens to verify [active_count, k]
            active_attention_mask: Attention mask for active sequences
            active_position_ids: Position IDs for active sequences
            active_indices: Indices of active sequences in batch
            relative_indices_tensor: Relative indices for KV cache
            current_seq_lens: Current sequence lengths per batch position
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            verify_stream: Optional CUDA stream for async execution
            draft_end_event: Event to wait for draft completion
            kv_cache_enabled: Whether KV cache is enabled
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (base_tokens, base_logits, verify_time_ms, verify_start_event, verify_end_event)
        """
        # Setup CUDA events for timing
        verify_start_event = None
        verify_end_event = None
        if (
            self.device == "cuda"
            and torch.cuda.is_available()
            and verify_stream is not None
        ):
            verify_start_event = torch.cuda.Event(enable_timing=True)
            verify_end_event = torch.cuda.Event(enable_timing=True)

        verify_start_wall = time.time()

        # Prepare base model KV cache
        base_past_kv = None
        base_current_seq_lens_for_model: Optional[List[int]] = None
        if kv_cache_enabled:
            base_past_kv = self.kv_cache_manager.get_base_past_kv(
                relative_indices_tensor
            )
            base_current_seq_lens_for_model = [
                (
                    self.kv_cache_manager.base_current_seq_lens[i]
                    if i < len(self.kv_cache_manager.base_current_seq_lens)
                    else current_seq_lens[active_indices[i]]
                )
                for i in range(len(active_indices))
            ]

        # Wait for draft to complete before appending
        if draft_end_event is not None:
            draft_end_event.synchronize()

        # Append draft tokens to input for parallel verification
        try:
            base_input_for_verify = active_input_ids
            if hasattr(self, "_active_input_ids_prepared"):
                base_input_for_verify = self._active_input_ids_prepared
        except (NameError, AttributeError):
            base_input_for_verify = active_input_ids

        if (
            draft_tokens is not None
            and draft_tokens.numel() > 0
            and draft_tokens.shape[1] > 0
        ):
            verify_input_ids = torch.cat([base_input_for_verify, draft_tokens], dim=1)
        else:
            verify_input_ids = base_input_for_verify

        # Update attention mask to include draft tokens
        verify_attention_mask = None
        if active_attention_mask is not None:
            if draft_tokens is not None and draft_tokens.numel() > 0:
                draft_mask = torch.ones(
                    (draft_tokens.shape[0], draft_tokens.shape[1]),
                    dtype=active_attention_mask.dtype,
                    device=active_attention_mask.device,
                )
                verify_attention_mask = torch.cat(
                    [active_attention_mask, draft_mask], dim=1
                )
            else:
                verify_attention_mask = active_attention_mask

        # Update position IDs
        verify_position_ids = None
        if active_position_ids is not None:
            if draft_tokens is not None and draft_tokens.numel() > 0:
                max_pos = (
                    active_position_ids.max().item()
                    if active_position_ids.numel() > 0
                    else 0
                )
                draft_positions = (
                    torch.arange(
                        1,
                        draft_tokens.shape[1] + 1,
                        dtype=active_position_ids.dtype,
                        device=active_position_ids.device,
                    )
                    .unsqueeze(0)
                    .expand(draft_tokens.shape[0], -1)
                    + max_pos
                )
                verify_position_ids = torch.cat(
                    [active_position_ids, draft_positions], dim=1
                )
            else:
                verify_position_ids = active_position_ids

        # Validate input before verification
        base_vocab_size = get_vocab_size(self.base_lm)
        if base_vocab_size is not None:
            verify_input_ids = validate_and_clamp_tokens(
                verify_input_ids, base_vocab_size, "base_input"
            )

        # CRITICAL: For verification, base model should use greedy (do_sample=False)
        # to match the argmax comparison in acceptance policy. The acceptance policy
        # compares draft tokens with base_logits.argmax(), so base must generate
        # argmax tokens (greedy) for consistency. The bonus token can still use sampling.
        verify_do_sample = False  # Always use greedy for verification consistency

        # Verify with base model
        if verify_stream is not None:
            if verify_start_event is not None:
                verify_start_event.record(verify_stream)

            with torch.cuda.stream(verify_stream):
                base_tokens, base_logits = self.base_lm.generate_tokens(
                    verify_input_ids,
                    max_new_tokens=1,  # Only generate bonus token
                    temperature=temperature,
                    do_sample=verify_do_sample,  # Greedy for verification consistency
                    past_key_values=base_past_kv,
                    attention_mask=verify_attention_mask,
                    position_ids=verify_position_ids,
                    current_seq_lens=base_current_seq_lens_for_model,
                    **kwargs,
                )

            if verify_end_event is not None:
                verify_end_event.record(verify_stream)
        else:
            # Sequential path
            base_tokens, base_logits = self.base_lm.generate_tokens(
                verify_input_ids,
                max_new_tokens=1,
                temperature=temperature,
                do_sample=verify_do_sample,  # Greedy for verification consistency
                past_key_values=base_past_kv,
                attention_mask=verify_attention_mask,
                position_ids=verify_position_ids,
                current_seq_lens=base_current_seq_lens_for_model,
                **kwargs,
            )

        # Update base KV cache
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
                if hasattr(self.base_lm._last_generated_kv, "past_key_values"):
                    base_new_kv = self.base_lm._last_generated_kv.past_key_values
                else:
                    base_new_kv = self.base_lm._last_generated_kv

            if base_new_kv is not None:
                if isinstance(base_new_kv, tuple):
                    self.kv_cache_manager.update_base_cache(
                        base_new_kv, active_indices, tokens_appended=None
                    )
                elif len(base_new_kv) > 0:
                    self.kv_cache_manager.update_base_cache(
                        tuple(base_new_kv), active_indices, tokens_appended=None
                    )

        # Calculate verify time
        if verify_start_event is not None and verify_end_event is not None:
            verify_time_ms = verify_start_event.elapsed_time(verify_end_event)
        else:
            verify_time_ms = (time.time() - verify_start_wall) * 1000

        return (
            base_tokens,
            base_logits,
            verify_time_ms,
            verify_start_event,
            verify_end_event,
        )


class AcceptanceHandler:
    """Handles token acceptance logic and duplication detection."""

    def __init__(
        self,
        policy: Any,
        tokenizer: Any,
        base_lm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the acceptance handler.

        Args:
            policy: Acceptance policy instance
            tokenizer: Tokenizer for EOS detection
            base_lm: Base language model for vocab size
            logger: Optional logger instance
        """
        self.policy = policy
        self.tokenizer = tokenizer
        self.base_lm = base_lm
        self.logger = logger or logging.getLogger(__name__)

    def apply_acceptance_policy(
        self,
        prompt_draft_tokens: torch.Tensor,
        prompt_base_tokens: torch.Tensor,
        prompt_draft_logits: torch.Tensor,
        prompt_base_logits: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Apply acceptance policy to determine how many draft tokens to accept.

        Args:
            prompt_draft_tokens: Draft tokens for this prompt [1, k]
            prompt_base_tokens: Base model tokens [1, k+1]
            prompt_draft_logits: Draft model logits [1, k, vocab_size]
            prompt_base_logits: Base model logits [1, k+1, vocab_size]

        Returns:
            Tuple of (accepted_len, policy_info)
        """
        accepted_len, policy_info = self.policy.accept_tokens(
            prompt_draft_tokens,
            prompt_base_tokens,
            prompt_draft_logits,
            prompt_base_logits,
        )
        return accepted_len, policy_info

    def extract_accepted_tokens(
        self,
        prompt_base_tokens: torch.Tensor,
        accepted_len: int,
        global_idx: int,
    ) -> torch.Tensor:
        """
        Extract accepted tokens from base model output.

        Args:
            prompt_base_tokens: Base model tokens [1, k+1]
            accepted_len: Number of tokens to accept
            global_idx: Global prompt index for error reporting

        Returns:
            Accepted tokens tensor [accepted_len]
        """
        if accepted_len > prompt_base_tokens.shape[1]:
            self.logger.error(
                f"CRITICAL: accepted_len ({accepted_len}) > base_tokens.shape[1] "
                f"({prompt_base_tokens.shape[1]}) for prompt {global_idx}"
            )
            accepted_len = prompt_base_tokens.shape[1]

        accepted_tokens_tensor = prompt_base_tokens[0, :accepted_len].clone()

        # Validate tokens
        base_vocab_size = get_vocab_size(self.base_lm)
        if base_vocab_size is not None:
            accepted_tokens_tensor = validate_and_clamp_tokens(
                accepted_tokens_tensor, base_vocab_size, "accepted_tokens"
            )

        return accepted_tokens_tensor

    def detect_duplication(
        self,
        accepted_tokens: List[int],
        generated_so_far: List[int],
        global_idx: int,
        step: int,
    ) -> List[int]:
        """
        Detect and remove duplicate tokens from accepted tokens.

        Args:
            accepted_tokens: List of accepted token IDs
            generated_so_far: List of previously generated tokens
            global_idx: Global prompt index
            step: Current step number

        Returns:
            Filtered accepted tokens with duplicates removed
        """
        if not accepted_tokens or not generated_so_far:
            return accepted_tokens

        # Check for phrase repetition (up to 30 tokens for better detection)
        max_check_len = min(30, len(generated_so_far), len(accepted_tokens))
        for check_len in range(max_check_len, 0, -1):
            generated_tail = generated_so_far[-check_len:]
            accepted_head = accepted_tokens[:check_len]
            if generated_tail == accepted_head:
                if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                    self.logger.warning(
                        f"CRITICAL: Detected {check_len}-token overlap "
                        f"for prompt {global_idx} at step {step}. Skipping duplicates."
                    )
                accepted_tokens = accepted_tokens[check_len:]
                break

        # Check for repeated single token pattern (improved detection)
        if accepted_tokens and generated_so_far:
            last_generated = generated_so_far[-1]
            if accepted_tokens[0] == last_generated:
                repeated_count = 1
                # Check up to 20 tokens for repetition (was 10)
                for i in range(1, min(20, len(accepted_tokens))):
                    if accepted_tokens[i] == last_generated:
                        repeated_count += 1
                    else:
                        break
                # Skip if 1+ repetitions (was >= 2, now >= 1 for stricter filtering)
                if repeated_count >= 1:
                    accepted_tokens = accepted_tokens[repeated_count:]
                    if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
                        self.logger.warning(
                            f"CRITICAL: Detected {repeated_count}-token repetition "
                            f"for prompt {global_idx} at step {step}. Skipping duplicates."
                        )

        # Additional check: detect short repetitive patterns (2-5 tokens)
        if accepted_tokens and len(generated_so_far) >= 2:
            for pattern_len in range(
                2, min(6, len(generated_so_far) + 1, len(accepted_tokens) + 1)
            ):
                generated_pattern = generated_so_far[-pattern_len:]
                # Check if this pattern repeats at the start of accepted tokens
                if len(accepted_tokens) >= pattern_len:
                    accepted_pattern = accepted_tokens[:pattern_len]
                    if generated_pattern == accepted_pattern:
                        # Check if it repeats multiple times
                        repeat_count = 1
                        for offset in range(
                            pattern_len,
                            len(accepted_tokens) - pattern_len + 1,
                            pattern_len,
                        ):
                            if (
                                accepted_tokens[offset : offset + pattern_len]
                                == generated_pattern
                            ):
                                repeat_count += 1
                            else:
                                break
                        if repeat_count >= 2:  # Pattern repeats 2+ times
                            skip_tokens = repeat_count * pattern_len
                            accepted_tokens = accepted_tokens[skip_tokens:]
                            if os.getenv("SPECDEC_DEBUG", "0").lower() in (
                                "1",
                                "true",
                                "yes",
                            ):
                                self.logger.warning(
                                    f"CRITICAL: Detected {repeat_count}x {pattern_len}-token pattern repetition "
                                    f"for prompt {global_idx} at step {step}. Skipping {skip_tokens} tokens."
                                )
                            break

        return accepted_tokens

    def sample_fallback_token(
        self,
        prompt_base_logits: torch.Tensor,
        temperature: float,
        do_sample: bool,
        sample_fn: Any,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample a fallback token when no draft tokens are accepted.

        Args:
            prompt_base_logits: Base model logits at position 0 [vocab_size]
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            sample_fn: Function to sample from logits (to avoid circular import)
            **kwargs: Additional generation parameters

        Returns:
            Fallback token tensor [1]
        """
        first_base_logits = prompt_base_logits[0, 0, :]  # [vocab_size]
        base_vocab_size = get_vocab_size(self.base_lm)

        top_p = kwargs.get("top_p", None)
        top_k = kwargs.get("top_k", None)

        fallback_token = sample_fn(
            logits=first_base_logits,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            vocab_size=base_vocab_size,
        )

        # Validate
        if base_vocab_size is not None:
            fallback_token = validate_and_clamp_tokens(
                fallback_token, base_vocab_size, "fallback_base"
            )

        return fallback_token


class RollbackHandler:
    """Handles zero-copy pointer rollback operations."""

    def __init__(
        self,
        kv_cache_manager: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the rollback handler.

        Args:
            kv_cache_manager: KV cache manager instance
            logger: Optional logger instance
        """
        self.kv_cache_manager = kv_cache_manager
        self.logger = logger or logging.getLogger(__name__)

    def update_sequence_pointers(
        self,
        idx_in_active: int,
        global_idx: int,
        original_len: int,
        accepted_len: int,
        current_seq_lens: List[int],
        kv_cache_enabled: bool = True,
    ) -> int:
        """
        Update sequence length pointers for zero-copy rollback.

        Args:
            idx_in_active: Index in active sequences
            global_idx: Global prompt index
            original_len: Original sequence length before draft generation
            accepted_len: Number of tokens accepted
            current_seq_lens: Current sequence lengths list (modified in place)
            kv_cache_enabled: Whether KV cache is enabled

        Returns:
            New sequence length after rollback
        """
        if not kv_cache_enabled:
            return original_len + accepted_len

        # FAST REWIND: Update pointer to reflect only accepted tokens
        new_base_seq_len = original_len + accepted_len

        # Update cache manager's pointer (this is the fast rewind)
        if idx_in_active < len(self.kv_cache_manager.base_current_seq_lens):
            self.kv_cache_manager.base_current_seq_lens[idx_in_active] = (
                new_base_seq_len
            )

        # Update tracking pointer
        current_seq_lens[global_idx] = new_base_seq_len

        return new_base_seq_len

    def sync_draft_cache_pointer(
        self,
        idx_in_active: int,
        original_len: int,
        accepted_len: int,
        enable_debug_prints: bool = False,
        step: int = 0,
        global_idx: int = 0,
    ) -> None:
        """
        Synchronize draft cache pointer to match base cache after bonus token.

        Args:
            idx_in_active: Index in active sequences
            original_len: Original sequence length
            accepted_len: Number of tokens accepted
            enable_debug_prints: Whether to enable debug prints
            step: Current step number
            global_idx: Global prompt index
        """
        if idx_in_active < len(self.kv_cache_manager.draft_current_seq_lens):
            # Draft cache should be at: original_len + accepted_len + 1 (bonus)
            new_draft_seq_len = original_len + accepted_len + 1
            self.kv_cache_manager.draft_current_seq_lens[idx_in_active] = (
                new_draft_seq_len
            )

            if enable_debug_prints and step <= 2:
                self.logger.debug(
                    f"[DRAFT_SYNC] Prompt {global_idx}: "
                    f"Appended bonus token to draft cache, new_draft_len={new_draft_seq_len}"
                )


class KVCacheHandler:
    """Handles KV cache management operations."""

    def __init__(
        self,
        kv_cache_manager: Any,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the KV cache handler.

        Args:
            kv_cache_manager: KV cache manager instance
            logger: Optional logger instance
        """
        self.kv_cache_manager = kv_cache_manager
        self.logger = logger or logging.getLogger(__name__)

    def update_sequence_lengths_dict(
        self,
        active_indices: List[int],
        global_sequence_ids: List[int],
        current_seq_lens: List[int],
        batch_active: List[bool],
    ) -> None:
        """
        Update sequence_lengths dict for compatibility.

        Args:
            active_indices: List of active sequence indices
            global_sequence_ids: Global sequence ID mapping
            current_seq_lens: Current sequence lengths
            batch_active: List of active flags
        """
        for idx_in_active, global_idx in enumerate(active_indices):
            if batch_active[global_idx]:
                global_id = global_sequence_ids[global_idx]
                self.kv_cache_manager.base_sequence_lengths[global_id] = (
                    current_seq_lens[global_idx]
                )

                if idx_in_active < len(self.kv_cache_manager.draft_current_seq_lens):
                    self.kv_cache_manager.draft_sequence_lengths[global_id] = (
                        self.kv_cache_manager.draft_current_seq_lens[idx_in_active]
                    )
                else:
                    self.kv_cache_manager.draft_sequence_lengths[global_id] = (
                        current_seq_lens[global_idx]
                    )
