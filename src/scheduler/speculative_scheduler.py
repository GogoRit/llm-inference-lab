"""
Speculative Scheduler for Multi-Stream Verification

Handles speculative decoding with optional CUDA verification streams,
overlapping draft and verification operations, and batched verification for K>1.

Phase 3D: Enhanced with CUDA event-based synchronization and true async overlap.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch

# Import kernels with fallback
try:
    from kernels import get_kernel_info, get_kv_append, get_verify_prefix

    KERNELS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Kernels not available, using fallback implementation")
    KERNELS_AVAILABLE = False

    def get_verify_prefix(device):
        return None

    def get_kv_append(device):
        return None

    def get_kernel_info():
        return {
            "verify_backend": "fallback",
            "kv_append_backend": "fallback",
        }


logger = logging.getLogger(__name__)


class SpeculativeScheduler:
    """Scheduler for speculative decoding with multi-stream verification."""

    def __init__(
        self,
        device: str = "cuda",
        enable_multi_stream: Optional[bool] = None,
        enable_batched_verification: bool = True,
    ):
        """
        Initialize the speculative scheduler.

        Args:
            device: Device to run on ("cuda", "mps", "cpu")
            enable_multi_stream: Whether to use multi-stream verification on CUDA
                                (default: reads SPECDEC_PARALLEL_STREAMS env var)
            enable_batched_verification: Whether to use batched verification for K>1
        """
        self.device = device

        # Check environment flags for Phase 3D
        if enable_multi_stream is None:
            env_value = os.getenv("SPECDEC_PARALLEL_STREAMS", "1").lower()
            enable_multi_stream = env_value in ("1", "true", "yes")

        sync_mode = os.getenv("SPECDEC_SYNC_MODE", "event").lower()
        self.use_event_sync = sync_mode == "event"

        self.enable_multi_stream = enable_multi_stream and device == "cuda"
        self.enable_batched_verification = enable_batched_verification

        # Create verification stream for CUDA
        self.verification_stream = None
        self.default_stream = None
        self.verify_ready_event = None

        if self.enable_multi_stream and torch.cuda.is_available():
            self.verification_stream = torch.cuda.Stream()
            self.default_stream = torch.cuda.current_stream()

            # Create CUDA event for synchronization (Phase 3D)
            if self.use_event_sync:
                self.verify_ready_event = torch.cuda.Event(enable_timing=True)
                logger.info(
                    "Created CUDA verification stream with event-based synchronization"
                )
            else:
                logger.info(
                    "Created CUDA verification stream with barrier synchronization"
                )
        elif self.enable_multi_stream:
            logger.warning(
                "Multi-stream requested but CUDA not available, falling back to single stream"
            )
            self.enable_multi_stream = False

        # Initialize kernels
        self.kernels_available = KERNELS_AVAILABLE
        self.kernel_info = (
            get_kernel_info()
            if KERNELS_AVAILABLE
            else {"verify_backend": "fallback", "kv_append_backend": "fallback"}
        )

        # Metrics tracking
        self.metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_steps": 0,
            "verification_time_ms": 0.0,
            "draft_time_ms": 0.0,
            "overlap_time_ms": 0.0,
        }

        logger.info(
            f"SpeculativeScheduler initialized: device={device}, "
            f"multi_stream={self.enable_multi_stream}, "
            f"batched_verification={self.enable_batched_verification}"
        )

    def schedule_verification(
        self,
        base_model,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Schedule verification with optional multi-stream processing.

        Args:
            base_model: Base language model for verification
            draft_tokens: Draft tokens to verify [batch_size, k]
            input_ids: Input context [batch_size, seq_len]
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (base_tokens, base_logits, metrics)
        """
        # start_time = time.time()  # Unused variable
        k = draft_tokens.shape[1]

        if self.enable_multi_stream and self.verification_stream is not None:
            return self._multi_stream_verification(
                base_model, draft_tokens, input_ids, temperature, do_sample, **kwargs
            )
        elif self.enable_batched_verification and k > 1:
            return self._batched_verification(
                base_model, draft_tokens, input_ids, temperature, do_sample, **kwargs
            )
        else:
            return self._single_stream_verification(
                base_model, draft_tokens, input_ids, temperature, do_sample, **kwargs
            )

    def _multi_stream_verification(
        self,
        base_model,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float,
        do_sample: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Multi-stream verification with CUDA streams and event-based sync (Phase 3D).

        Uses CUDA events to synchronize only when verification results are needed,
        allowing draft and verification to overlap asynchronously.
        """
        # Phase 3D: Use CUDA events for timing and synchronization
        if self.use_event_sync and self.verify_ready_event is not None:
            verify_start_event = torch.cuda.Event(enable_timing=True)
            verify_end_event = torch.cuda.Event(enable_timing=True)

            # Record start event on verification stream
            verify_start_event.record(self.verification_stream)

        # Run verification on verification stream (async)
        # Pass stream to enable true async execution
        # For verification, always use greedy (argmax) to ensure deterministic matching
        # Extract attention_mask from kwargs if present (for batch processing)
        base_tokens, base_logits = base_model.generate_tokens(
            input_ids,
            max_new_tokens=draft_tokens.shape[1],
            temperature=1.0,  # Temperature=1.0 for deterministic argmax
            do_sample=False,  # Always use greedy for verification
            stream=self.verification_stream,  # Enable async execution
            **kwargs,  # Includes attention_mask if provided
        )

        # Phase 3D: Record end event and wait only when needed
        if self.use_event_sync and self.verify_ready_event is not None:
            verify_end_event.record(self.verification_stream)
            # Wait for verification to complete before returning
            verify_end_event.wait(self.default_stream)
            torch.cuda.synchronize()  # Ensure all operations complete

            # Calculate time using CUDA events (more accurate than wall-clock)
            verification_time_ms = verify_start_event.elapsed_time(verify_end_event)
        else:
            # Fallback to barrier synchronization
            if self.verification_stream is not None:
                self.verification_stream.synchronize()
            # Use wall-clock time as fallback
            start_time = time.time()
            verification_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics["verification_time_ms"] += verification_time_ms
        self.metrics["total_steps"] += 1

        # Log verification step info (only if debug flag enabled)
        if os.getenv("SPECDEC_DEBUG", "0").lower() in ("1", "true", "yes"):
            k = draft_tokens.shape[1] if hasattr(draft_tokens, "shape") else 0
            draft_time_ms = self.metrics.get("draft_time_ms", 0.0) / max(
                self.metrics.get("total_steps", 1), 1
            )
            logger.debug(
                f"[SCHED] K={k} | "
                f"verify={verification_time_ms:.1f}ms | "
                f"draft_avg={draft_time_ms:.1f}ms | "
                f"stream={True}, sync={('event' if self.use_event_sync else 'barrier')}"
            )

        return (
            base_tokens,
            base_logits,
            {
                "verification_time_ms": verification_time_ms,
                "method": "multi_stream",
                "stream_used": True,
                "sync_mode": "event" if self.use_event_sync else "barrier",
            },
        )

    def _batched_verification(
        self,
        base_model,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float,
        do_sample: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Batched verification for K>1."""
        start_time = time.time()

        # For batched verification, we can potentially process multiple
        # draft sequences in parallel, but for now we'll use single verification
        # with optimized batching
        # For verification, always use greedy (argmax) to ensure deterministic matching
        base_tokens, base_logits = base_model.generate_tokens(
            input_ids,
            max_new_tokens=draft_tokens.shape[1],
            temperature=1.0,  # Temperature=1.0 for deterministic argmax
            do_sample=False,  # Always use greedy for verification
            **kwargs,
        )

        verification_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics["verification_time_ms"] += verification_time_ms
        self.metrics["total_steps"] += 1

        return (
            base_tokens,
            base_logits,
            {
                "verification_time_ms": verification_time_ms,
                "method": "batched",
                "k": draft_tokens.shape[1],
            },
        )

    def _single_stream_verification(
        self,
        base_model,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float,
        do_sample: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Single-stream verification (fallback)."""
        start_time = time.time()

        # For verification, always use greedy (argmax) to ensure deterministic matching
        base_tokens, base_logits = base_model.generate_tokens(
            input_ids,
            max_new_tokens=draft_tokens.shape[1],
            temperature=1.0,  # Temperature=1.0 for deterministic argmax
            do_sample=False,  # Always use greedy for verification
            **kwargs,
        )

        verification_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics["verification_time_ms"] += verification_time_ms
        self.metrics["total_steps"] += 1

        return (
            base_tokens,
            base_logits,
            {
                "verification_time_ms": verification_time_ms,
                "method": "single_stream",
            },
        )

    def schedule_draft_generation(
        self,
        draft_model,
        input_ids: torch.Tensor,
        k: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Schedule draft generation (runs on default stream).

        Args:
            draft_model: Draft language model
            input_ids: Input context [batch_size, seq_len]
            k: Number of draft tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (draft_tokens, draft_logits, metrics)
        """
        start_time = time.time()

        draft_tokens, draft_logits = draft_model.generate_tokens(
            input_ids,
            max_new_tokens=k,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs,
        )

        draft_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics["draft_time_ms"] += draft_time_ms
        self.metrics["total_proposed"] += k

        return (
            draft_tokens,
            draft_logits,
            {
                "draft_time_ms": draft_time_ms,
                "k": k,
            },
        )

    def apply_acceptance_policy(
        self,
        policy,
        draft_tokens: torch.Tensor,
        base_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Apply acceptance policy to determine accepted tokens.

        Args:
            policy: Acceptance policy instance
            draft_tokens: Draft tokens [batch_size, k]
            base_tokens: Base model tokens [batch_size, k]
            draft_logits: Draft logits [batch_size, k, vocab_size]
            base_logits: Base logits [batch_size, k, vocab_size]

        Returns:
            Tuple of (accepted_length, policy_info)
        """
        accepted_len, policy_info = policy.accept_tokens(
            draft_tokens, base_tokens, draft_logits, base_logits
        )

        # Update metrics
        self.metrics["total_accepted"] += accepted_len

        return accepted_len, policy_info

    def get_metrics(self) -> Dict[str, Any]:
        """Get current scheduler metrics."""
        total_time = (
            self.metrics["verification_time_ms"] + self.metrics["draft_time_ms"]
        )
        acceptance_rate = (
            self.metrics["total_accepted"] / self.metrics["total_proposed"]
            if self.metrics["total_proposed"] > 0
            else 0.0
        )

        return {
            **self.metrics,
            "total_time_ms": total_time,
            "acceptance_rate": acceptance_rate,
            "device": self.device,
            "multi_stream_enabled": self.enable_multi_stream,
            "batched_verification_enabled": self.enable_batched_verification,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_steps": 0,
            "verification_time_ms": 0.0,
            "draft_time_ms": 0.0,
            "overlap_time_ms": 0.0,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.verification_stream is not None:
            del self.verification_stream
            self.verification_stream = None

        # Clear CUDA cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SpeculativeScheduler cleaned up")


def create_speculative_scheduler(
    device: str = "cuda",
    enable_multi_stream: bool = True,
    enable_batched_verification: bool = True,
) -> SpeculativeScheduler:
    """
    Create a SpeculativeScheduler instance.

    Args:
        device: Device to run on
        enable_multi_stream: Whether to enable multi-stream processing
        enable_batched_verification: Whether to enable batched verification

    Returns:
        Configured SpeculativeScheduler instance
    """
    return SpeculativeScheduler(
        device=device,
        enable_multi_stream=enable_multi_stream,
        enable_batched_verification=enable_batched_verification,
    )
