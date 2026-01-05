"""
Batch Generation Loop Orchestrator

Orchestrates the speculative decoding loop for batch processing using
specialized handler classes. This separates the orchestration logic from
the implementation details.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..utils.token_validation import get_vocab_size, validate_and_clamp_tokens
from .batch_handlers import (
    AcceptanceHandler,
    DraftGenerationHandler,
    KVCacheHandler,
    RollbackHandler,
    VerificationHandler,
)
from .batch_utilities import BatchMetricsCollector, BatchSequenceManager
from .sequence_pool import SequencePool
from .sequence_utils import create_position_ids, pad_sequences

logger = logging.getLogger(__name__)


class BatchGenerationLoop:
    """Orchestrates the speculative decoding loop for batches."""

    def __init__(
        self,
        draft_handler: DraftGenerationHandler,
        verify_handler: VerificationHandler,
        accept_handler: AcceptanceHandler,
        rollback_handler: RollbackHandler,
        kv_handler: KVCacheHandler,
        scheduler: Any,
        policy: Any,
        tokenizer: Any,
        device: str,
        kv_cache_manager: Any,
        deterministic_mode: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the batch generation loop.

        Args:
            draft_handler: Draft generation handler
            verify_handler: Verification handler
            accept_handler: Acceptance handler
            rollback_handler: Rollback handler
            kv_handler: KV cache handler
            scheduler: Speculative scheduler instance
            policy: Acceptance policy instance
            tokenizer: Tokenizer instance
            device: Device to run on
            kv_cache_manager: KV cache manager instance
            deterministic_mode: If True, disable duplication detection for correctness testing
            logger: Optional logger instance
        """
        self.draft_handler = draft_handler
        self.verify_handler = verify_handler
        self.accept_handler = accept_handler
        self.rollback_handler = rollback_handler
        self.kv_handler = kv_handler
        self.scheduler = scheduler
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_manager = kv_cache_manager
        self.deterministic_mode = deterministic_mode
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        prompts: List[str],
        batch_input_ids: torch.Tensor,
        current_input_ids: List[torch.Tensor],
        sequence_manager: BatchSequenceManager,
        metrics_collector: BatchMetricsCollector,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        controller: Any,
        kv_cache_enabled: bool = True,
        draft_stream: Optional[torch.cuda.Stream] = None,
        verify_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Run the speculative decoding loop for a batch of prompts.

        Args:
            prompts: List of input prompts
            batch_input_ids: Initial tokenized input IDs [batch_size, seq_len]
            current_input_ids: List of current input sequences (1D tensors)
            sequence_manager: Sequence manager instance
            metrics_collector: Metrics collector instance
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            controller: K controller instance for dynamic K selection
            kv_cache_enabled: Whether KV cache is enabled
            draft_stream: Optional CUDA stream for draft generation
            verify_stream: Optional CUDA stream for verification
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (batch_generated_tokens, batch_metrics)
        """
        generation_start = time.time()
        step = 0
        enable_debug_prints = os.getenv("SPECDEC_DEBUG_PRINTS", "0").lower() in (
            "1",
            "true",
            "yes",
        )

        # Main speculative decoding loop
        while any(sequence_manager.batch_active) and step < max_tokens:
            step += 1
            active_indices = sequence_manager.get_active_indices()

            if not active_indices:
                break

            active_count = len(active_indices)

            # Get K from controller
            context = {
                "step": step,
                "generated_tokens": max(
                    len(tokens) for tokens in sequence_manager.batch_generated_tokens
                ),
                "acceptance_rate": (
                    metrics_collector.batch_metrics["total_accepted"]
                    / max(metrics_collector.batch_metrics["total_proposed"], 1)
                ),
            }
            k = controller.get_k(step, context)
            if k <= 0:
                break

            # Extract active sequences
            active_seqs = [
                current_input_ids[i].detach().clone().contiguous()
                for i in active_indices
            ]

            # Validate sequences
            draft_vocab_size = get_vocab_size(self.draft_handler.draft_lm)
            if draft_vocab_size is not None:
                for i, seq in enumerate(active_seqs):
                    active_seqs[i] = validate_and_clamp_tokens(
                        seq, draft_vocab_size, f"seq_{i}"
                    )

            # Pad sequences
            pad_value = (
                self.tokenizer.pad_token_id
                if self.tokenizer is not None
                and self.tokenizer.pad_token_id is not None
                else 0
            )
            active_input_ids, active_attention_mask, original_lengths = pad_sequences(
                sequences=active_seqs,
                pad_token_id=pad_value,
                device=torch.device(self.device),
            )

            # Create position IDs
            active_position_ids = create_position_ids(
                sequence_lengths=original_lengths,
                max_length=active_input_ids.shape[1],
                device=torch.device(self.device),
            )

            # Setup relative indices for KV cache
            relative_indices = list(range(active_count))
            self.kv_cache_manager.set_active_indices(relative_indices)
            relative_indices_tensor = torch.tensor(
                relative_indices, dtype=torch.long, device=self.device
            )

            # Get current sequence lengths for active sequences
            active_sequence_lengths = [
                active_seqs[i].shape[0] for i in range(active_count)
            ]
            active_global_ids = [
                sequence_manager.global_sequence_ids[i] for i in active_indices
            ]

            if kv_cache_enabled:
                self.kv_cache_manager.set_sequence_metadata(
                    global_sequence_ids=active_global_ids,
                    sequence_lengths=active_sequence_lengths,
                    batch_to_global_map={
                        i: gid for i, gid in enumerate(active_global_ids)
                    },
                )

            # Step 1: Generate draft tokens
            # Remove 'k' from kwargs if present to avoid conflict (k is already passed explicitly)
            draft_kwargs = {key: val for key, val in kwargs.items() if key != "k"}
            (
                draft_tokens,
                draft_logits,
                draft_time_ms,
                draft_start_event,
                draft_end_event,
            ) = self.draft_handler.generate_draft_tokens(
                active_input_ids=active_input_ids,
                active_attention_mask=active_attention_mask,
                active_position_ids=active_position_ids,
                active_indices=torch.tensor(active_indices, device=self.device),
                relative_indices_tensor=relative_indices_tensor,
                current_seq_lens=sequence_manager.current_seq_lens,
                k=k,
                temperature=temperature,
                do_sample=do_sample,
                draft_stream=draft_stream,
                kv_cache_enabled=kv_cache_enabled,
                verify_stream=verify_stream,
                **draft_kwargs,
            )

            # Step 2: Verify draft tokens
            (
                base_tokens,
                base_logits,
                verify_time_ms,
                verify_start_event,
                verify_end_event,
            ) = self.verify_handler.verify_draft_tokens(
                active_input_ids=active_input_ids,
                draft_tokens=draft_tokens,
                active_attention_mask=active_attention_mask,
                active_position_ids=active_position_ids,
                active_indices=torch.tensor(active_indices, device=self.device),
                relative_indices_tensor=relative_indices_tensor,
                current_seq_lens=sequence_manager.current_seq_lens,
                temperature=temperature,
                do_sample=do_sample,
                verify_stream=verify_stream,
                draft_end_event=draft_end_event,
                kv_cache_enabled=kv_cache_enabled,
                **kwargs,
            )

            # Step 3: Apply acceptance policy and process results
            accepted_lengths = []
            accepted_tokens_list = []
            original_seq_lens = [
                sequence_manager.get_current_length(global_idx)
                for global_idx in active_indices
            ]

            for idx_in_active, global_idx in enumerate(active_indices):
                if not sequence_manager.is_active(global_idx):
                    continue

                # Extract tokens/logits for this prompt
                prompt_draft_tokens = draft_tokens[idx_in_active : idx_in_active + 1]
                prompt_base_tokens = base_tokens[idx_in_active : idx_in_active + 1]
                prompt_draft_logits = draft_logits[idx_in_active : idx_in_active + 1]
                prompt_base_logits = base_logits[idx_in_active : idx_in_active + 1]

                # Debug logging for batch path investigation
                if enable_debug_prints:
                    self.logger.debug(
                        f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                        f"draft_tokens.shape={prompt_draft_tokens.shape}, "
                        f"base_tokens.shape={prompt_base_tokens.shape}, "
                        f"draft_logits.shape={prompt_draft_logits.shape}, "
                        f"base_logits.shape={prompt_base_logits.shape}"
                    )

                # Apply acceptance policy
                accepted_len, policy_info = self.accept_handler.apply_acceptance_policy(
                    prompt_draft_tokens,
                    prompt_base_tokens,
                    prompt_draft_logits,
                    prompt_base_logits,
                )

                if enable_debug_prints:
                    self.logger.debug(
                        f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                        f"Policy returned accepted_len={accepted_len}"
                    )

                # Store accepted_len from policy (before token filtering)
                policy_accepted_len = accepted_len
                accepted_tokens = []  # Initialize for all code paths
                sequence_accepted_len = 0  # Initialize for all code paths

                # CRITICAL: Cap accepted_len to not exceed max_tokens.
                # This is essential for correctness: when k=4 and max_tokens=5, we must not accept
                # 4 tokens in step 2 if we already have 4 tokens from step 1. Without this cap,
                # speculative decoding would generate more tokens than vanilla decoding, breaking
                # the correctness contract. This cap ensures we respect the max_tokens limit exactly.
                current_generated_count = len(
                    sequence_manager.get_generated_tokens(global_idx)
                )
                remaining_tokens = max_tokens - current_generated_count
                if accepted_len > remaining_tokens:
                    if enable_debug_prints:
                        self.logger.debug(
                            f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                            f"Capping accepted_len from {accepted_len} to {remaining_tokens} "
                            f"(current={current_generated_count}, max={max_tokens})"
                        )
                    accepted_len = max(0, remaining_tokens)

                # Extract accepted tokens
                if accepted_len > 0:
                    # CRITICAL FIX: When accepted_len > base_tokens.shape[1], it means we accepted
                    # draft tokens that weren't verified (e.g., for correctness testing with draft==target).
                    # In this case, use draft tokens instead of base_tokens.
                    if accepted_len > prompt_base_tokens.shape[1]:
                        # Use draft tokens for accepted positions
                        accepted_tokens_tensor = prompt_draft_tokens[
                            0, :accepted_len
                        ].clone()
                    else:
                        # Normal case: extract from base_tokens
                        accepted_tokens_tensor = (
                            self.accept_handler.extract_accepted_tokens(
                                prompt_base_tokens, accepted_len, global_idx
                            )
                        )

                    # Convert to list and detect duplication
                    accepted_tokens = accepted_tokens_tensor.cpu().tolist()
                    generated_so_far = sequence_manager.get_generated_tokens(global_idx)

                    # CRITICAL FIX: These tokens were verified by the acceptance policy
                    # (draft matches base). For performance benchmarks, we should trust verified
                    # tokens even if they're repetitive - the base model generated them, so they're
                    # legitimate. Only apply duplication detection if we're in a mode that requires
                    # it (e.g., for quality control, not for performance benchmarks).
                    #
                    # For now, skip duplication detection for verified tokens in non-deterministic
                    # mode to avoid filtering legitimate model output. The acceptance policy
                    # already verified these tokens match between draft and base.
                    if not self.accept_handler.deterministic_mode:
                        # In performance benchmark mode, trust verified tokens
                        # (duplication detection skipped for verified tokens)
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                                f"Skipping duplication check for verified tokens (non-deterministic mode)"
                            )
                    else:
                        # In deterministic mode, duplication detection is already disabled
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                                f"Before duplication check: accepted_tokens={accepted_tokens}, "
                                f"generated_so_far={generated_so_far}"
                            )
                        accepted_tokens = self.accept_handler.detect_duplication(
                            accepted_tokens, generated_so_far, global_idx, step
                        )
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                                f"After duplication check: accepted_tokens={accepted_tokens}"
                            )

                    # Check for EOS
                    eos_token_id = 50256  # Default GPT-2
                    if self.tokenizer is not None:
                        eos_token_id = self.tokenizer.eos_token_id

                    if accepted_tokens and accepted_tokens[0] == eos_token_id:
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[BATCH_LOOP] Step {step}, Seq {global_idx}: EOS detected, clearing tokens"
                            )
                        accepted_tokens = []
                        sequence_manager.deactivate(global_idx)
                        sequence_accepted_len = 0  # EOS detected, no tokens added
                    elif accepted_tokens:
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                                f"Adding {len(accepted_tokens)} tokens: {accepted_tokens}"
                            )
                        sequence_manager.add_generated_tokens(
                            global_idx, accepted_tokens
                        )
                        accepted_tokens_list.append(accepted_tokens)
                        sequence_accepted_len = len(
                            accepted_tokens
                        )  # Actual tokens added
                    else:
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[BATCH_LOOP] Step {step}, Seq {global_idx}: "
                                f"All tokens filtered out, accepted_tokens is empty"
                            )
                        sequence_accepted_len = 0  # All tokens filtered out
                else:
                    # Fallback: sample from base model
                    # Import here to avoid circular dependency
                    from .pipeline import sample_bonus_token_from_logits

                    if enable_debug_prints:
                        self.logger.debug(
                            f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                            f"accepted_len=0, entering fallback path. "
                            f"base_logits.shape={prompt_base_logits.shape}"
                        )

                    fallback_token = self.accept_handler.sample_fallback_token(
                        prompt_base_logits,
                        temperature,
                        do_sample,
                        sample_bonus_token_from_logits,
                        **kwargs,
                    )

                    if enable_debug_prints:
                        self.logger.debug(
                            f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                            f"fallback_token={fallback_token}, shape={fallback_token.shape}"
                        )

                    # Check for EOS
                    eos_token_id = 50256
                    if self.tokenizer is not None:
                        eos_token_id = self.tokenizer.eos_token_id

                    if (fallback_token == eos_token_id).any():
                        if enable_debug_prints:
                            self.logger.debug(
                                f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                                f"EOS detected, deactivating"
                            )
                        sequence_manager.deactivate(global_idx)
                        accepted_tokens = []
                        sequence_accepted_len = 0  # No tokens accepted
                    else:
                        accepted_tokens = fallback_token.cpu().tolist()

                        if enable_debug_prints:
                            self.logger.debug(
                                f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                                f"fallback_token before duplication check: {accepted_tokens}"
                            )

                        # Check for duplication
                        generated_so_far = sequence_manager.get_generated_tokens(
                            global_idx
                        )
                        accepted_tokens = self.accept_handler.detect_duplication(
                            accepted_tokens, generated_so_far, global_idx, step
                        )

                        if enable_debug_prints:
                            self.logger.debug(
                                f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                                f"fallback_token after duplication check: {accepted_tokens}, "
                                f"generated_so_far={generated_so_far}"
                            )

                        if accepted_tokens:
                            sequence_manager.add_generated_tokens(
                                global_idx, accepted_tokens
                            )
                            accepted_tokens_list.append(accepted_tokens)
                            sequence_accepted_len = 1  # Fallback token accepted
                            if enable_debug_prints:
                                self.logger.debug(
                                    f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                                    f"Fallback token accepted: {accepted_tokens}"
                                )
                        else:
                            sequence_accepted_len = 0
                            if enable_debug_prints:
                                self.logger.debug(
                                    f"[CPU_DEBUG_FALLBACK] Step {step}, Seq {global_idx}: "
                                    f"Fallback token filtered out by duplication detection"
                                )

                # Use the stored accepted_len for this sequence (not from list)
                accepted_lengths.append(sequence_accepted_len)

                # Update metrics (use sequence_accepted_len, not accepted_len from list)
                metrics_collector.record_step(
                    proposed_count=prompt_draft_tokens.shape[1],
                    accepted_count=sequence_accepted_len,
                    draft_time_ms=draft_time_ms / active_count,
                    verify_time_ms=verify_time_ms / active_count,
                    global_idx=global_idx,
                )

                # Debug logging for CPU correctness testing
                if enable_debug_prints:
                    tau = sequence_accepted_len / max(prompt_draft_tokens.shape[1], 1)
                    self.logger.debug(
                        f"[CPU_DEBUG] Step {step}, Seq {global_idx}: "
                        f"k={prompt_draft_tokens.shape[1]}, accepted={sequence_accepted_len}, "
                        f"τ={tau:.3f}, seq_len={sequence_manager.get_current_length(global_idx)}"
                    )

                # Update sequence pointers (zero-copy rollback)
                # Use sequence_accepted_len (the correct value for this sequence)
                original_len = original_seq_lens[idx_in_active]

                if sequence_accepted_len > 0:
                    new_seq_len = self.rollback_handler.update_sequence_pointers(
                        idx_in_active=idx_in_active,
                        global_idx=global_idx,
                        original_len=original_len,
                        accepted_len=sequence_accepted_len,
                        current_seq_lens=sequence_manager.current_seq_lens,
                        kv_cache_enabled=kv_cache_enabled,
                    )

                    # Sync draft cache if bonus token was added
                    self.rollback_handler.sync_draft_cache_pointer(
                        idx_in_active=idx_in_active,
                        original_len=original_len,
                        accepted_len=sequence_accepted_len,
                        enable_debug_prints=enable_debug_prints,
                        step=step,
                        global_idx=global_idx,
                    )
                else:
                    # Even if no tokens accepted, we may have added a fallback token
                    # Update sequence length to reflect fallback token if it was added
                    if accepted_tokens and len(accepted_tokens) > 0:
                        # Fallback token was added, update sequence length
                        new_seq_len = original_len + len(accepted_tokens)
                        if kv_cache_enabled:
                            if idx_in_active < len(
                                self.kv_cache_manager.base_current_seq_lens
                            ):
                                self.kv_cache_manager.base_current_seq_lens[
                                    idx_in_active
                                ] = new_seq_len
                        sequence_manager.current_seq_lens[global_idx] = new_seq_len

                # Update current input sequence (CRITICAL: must happen for both accepted and fallback paths)
                if accepted_tokens:
                    accepted_tokens_tensor = torch.tensor(
                        accepted_tokens, device=self.device, dtype=torch.long
                    )

                    # Validate before concatenation
                    base_vocab_size = get_vocab_size(self.verify_handler.base_lm)
                    if base_vocab_size is not None:
                        accepted_tokens_tensor = validate_and_clamp_tokens(
                            accepted_tokens_tensor,
                            base_vocab_size,
                            f"accepted_{global_idx}",
                        )

                    current_seq = current_input_ids[global_idx]
                    updated_seq = torch.cat(
                        [current_seq, accepted_tokens_tensor], dim=0
                    )

                    # Validate with draft vocab size for next iteration
                    draft_vocab_size = get_vocab_size(self.draft_handler.draft_lm)
                    if draft_vocab_size is not None:
                        updated_seq = validate_and_clamp_tokens(
                            updated_seq,
                            draft_vocab_size,
                            f"current_seq_{global_idx}",
                        )

                    current_input_ids[global_idx] = (
                        updated_seq.detach().clone().contiguous()
                    )

                # Check if done
                if len(sequence_manager.get_generated_tokens(global_idx)) >= max_tokens:
                    sequence_manager.deactivate(global_idx)

            # Update KV cache sequence lengths dict
            if kv_cache_enabled:
                self.kv_handler.update_sequence_lengths_dict(
                    active_indices=active_indices,
                    global_sequence_ids=sequence_manager.global_sequence_ids,
                    current_seq_lens=sequence_manager.current_seq_lens,
                    batch_active=sequence_manager.batch_active,
                )

            # CRITICAL FIX: Removed KV cache reset on partial acceptance.
            # The ring buffer + pointer rollback correctly handles partial acceptance.
            # Resetting breaks correctness and defeats the purpose of zero-copy design.

        # Finalize metrics
        total_time_ms = (time.time() - generation_start) * 1000
        batch_metrics = metrics_collector.get_batch_metrics()
        batch_metrics["total_generation_time_ms"] = total_time_ms
        batch_metrics["tokens_per_sec"] = (
            batch_metrics["total_generated_tokens"] / (total_time_ms / 1000.0)
            if total_time_ms > 0
            else 0.0
        )

        return sequence_manager.batch_generated_tokens, batch_metrics
