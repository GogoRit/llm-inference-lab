"""
Batch Processing Utilities

Utility classes for managing sequences and metrics in batch processing.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class BatchSequenceManager:
    """Manages sequence tracking and active batch state."""

    def __init__(
        self,
        batch_size: int,
        initial_sequence_lengths: List[int],
        global_sequence_ids: List[int],
    ):
        """
        Initialize the sequence manager.

        Args:
            batch_size: Total batch size
            initial_sequence_lengths: Initial sequence lengths for each prompt
            global_sequence_ids: Global sequence ID mapping
        """
        self.batch_size = batch_size
        self.current_seq_lens = initial_sequence_lengths.copy()
        self.global_sequence_ids = global_sequence_ids
        self.batch_active = [True] * batch_size
        self.batch_generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]

    def get_active_indices(self) -> List[int]:
        """Get list of active sequence indices."""
        return [i for i in range(self.batch_size) if self.batch_active[i]]

    def deactivate(self, global_idx: int) -> None:
        """Mark a sequence as inactive."""
        if 0 <= global_idx < self.batch_size:
            self.batch_active[global_idx] = False

    def is_active(self, global_idx: int) -> bool:
        """Check if a sequence is still active."""
        return 0 <= global_idx < self.batch_size and self.batch_active[global_idx]

    def update_sequence_length(self, global_idx: int, new_length: int) -> None:
        """Update sequence length for a specific prompt."""
        if 0 <= global_idx < len(self.current_seq_lens):
            self.current_seq_lens[global_idx] = new_length

    def add_generated_tokens(self, global_idx: int, tokens: List[int]) -> None:
        """Add generated tokens to a sequence."""
        if 0 <= global_idx < len(self.batch_generated_tokens):
            self.batch_generated_tokens[global_idx].extend(tokens)

    def get_generated_tokens(self, global_idx: int) -> List[int]:
        """Get generated tokens for a sequence."""
        if 0 <= global_idx < len(self.batch_generated_tokens):
            return self.batch_generated_tokens[global_idx]
        return []

    def get_current_length(self, global_idx: int) -> int:
        """Get current sequence length for a prompt."""
        if 0 <= global_idx < len(self.current_seq_lens):
            return self.current_seq_lens[global_idx]
        return 0

    def get_active_count(self) -> int:
        """Get number of active sequences."""
        return sum(1 for active in self.batch_active if active)


class BatchMetricsCollector:
    """Collects and aggregates metrics during batch processing."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.batch_metrics = {
            "total_proposed": 0,
            "total_accepted": 0,
            "total_generated_tokens": 0,
            "total_steps": 0,
            "total_draft_time_ms": 0.0,
            "total_verification_time_ms": 0.0,
            "total_generation_time_ms": 0.0,
        }
        self.per_prompt_proposed_counts: List[int] = []
        self.per_prompt_accepted_counts: List[int] = []

    def initialize_per_prompt_tracking(self, batch_size: int) -> None:
        """Initialize per-prompt metric tracking."""
        self.per_prompt_proposed_counts = [0] * batch_size
        self.per_prompt_accepted_counts = [0] * batch_size

    def record_step(
        self,
        proposed_count: int,
        accepted_count: int,
        draft_time_ms: float,
        verify_time_ms: float,
        global_idx: Optional[int] = None,
    ) -> None:
        """
        Record metrics for a single step.

        Args:
            proposed_count: Number of tokens proposed
            accepted_count: Number of tokens accepted
            draft_time_ms: Draft generation time in milliseconds
            verify_time_ms: Verification time in milliseconds
            global_idx: Optional global prompt index for per-prompt tracking
        """
        self.batch_metrics["total_proposed"] += proposed_count
        self.batch_metrics["total_accepted"] += accepted_count
        self.batch_metrics["total_generated_tokens"] += accepted_count
        self.batch_metrics["total_steps"] += 1
        self.batch_metrics["total_draft_time_ms"] += draft_time_ms
        self.batch_metrics["total_verification_time_ms"] += verify_time_ms

        if global_idx is not None:
            if 0 <= global_idx < len(self.per_prompt_proposed_counts):
                self.per_prompt_proposed_counts[global_idx] += proposed_count
            if 0 <= global_idx < len(self.per_prompt_accepted_counts):
                self.per_prompt_accepted_counts[global_idx] += accepted_count

    def get_batch_metrics(self) -> Dict[str, Any]:
        """Get aggregated batch metrics."""
        return self.batch_metrics.copy()

    def get_per_prompt_metrics(self) -> Dict[str, List[int]]:
        """Get per-prompt metrics."""
        return {
            "proposed": self.per_prompt_proposed_counts.copy(),
            "accepted": self.per_prompt_accepted_counts.copy(),
        }
