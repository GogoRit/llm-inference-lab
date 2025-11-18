"""
Sequence Pool for EXSPEC-style Length-Aware Scheduling

Groups sequences by length to minimize padding overhead by processing
same-length sequences together without padding.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SequencePool:
    """
    Pool of sequences grouped by length for efficient batching.

    Maintains sequences keyed by their current length and provides
    methods to extract same-length groups for processing.
    """

    def __init__(
        self,
        max_group_size: Optional[int] = None,
        min_group_size: int = 1,
    ):
        """
        Initialize sequence pool.

        Args:
            max_group_size: Maximum number of sequences per group (None = no limit)
            min_group_size: Minimum group size before falling back to mixed batches
        """
        self.max_group_size = max_group_size
        self.min_group_size = min_group_size

        # Map: length -> list of (global_id, sequence_tensor)
        self._length_groups: Dict[int, List[Tuple[int, torch.Tensor]]] = defaultdict(
            list
        )

        # Map: global_id -> (length, status)
        self._sequence_metadata: Dict[int, Tuple[int, bool]] = {}

        # Statistics
        self._stats: Dict[str, float] = {
            "total_groups_formed": 0.0,
            "same_length_tokens": 0.0,
            "mixed_length_tokens": 0.0,
            "total_tokens_processed": 0.0,
        }

    def add_sequence(
        self,
        global_id: int,
        sequence: torch.Tensor,
        is_active: bool = True,
    ) -> None:
        """
        Add or update a sequence in the pool.

        Args:
            global_id: Global sequence identifier
            sequence: 1D tensor of tokens
            is_active: Whether sequence is still generating
        """
        length = sequence.shape[0]

        # Remove from old length group if exists
        if global_id in self._sequence_metadata:
            old_length, _ = self._sequence_metadata[global_id]
            if old_length != length:
                # Remove from old group
                self._length_groups[old_length] = [
                    (gid, seq)
                    for gid, seq in self._length_groups[old_length]
                    if gid != global_id
                ]
                # Clean up empty groups
                if not self._length_groups[old_length]:
                    del self._length_groups[old_length]

        # Add to new length group
        if is_active:
            self._length_groups[length].append((global_id, sequence))

        # Update metadata
        self._sequence_metadata[global_id] = (length, is_active)

    def remove_sequence(self, global_id: int) -> None:
        """Remove a sequence from the pool."""
        if global_id not in self._sequence_metadata:
            return

        length, _ = self._sequence_metadata[global_id]

        # Remove from length group
        self._length_groups[length] = [
            (gid, seq) for gid, seq in self._length_groups[length] if gid != global_id
        ]

        # Clean up empty groups
        if not self._length_groups[length]:
            del self._length_groups[length]

        # Remove metadata
        del self._sequence_metadata[global_id]

    def get_same_length_group(
        self,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> Optional[Tuple[List[int], List[torch.Tensor], int]]:
        """
        Get a group of sequences with the same length.

        Returns the largest available same-length group, or None if none available.

        Args:
            min_size: Minimum group size (uses self.min_group_size if None)
            max_size: Maximum group size (uses self.max_group_size if None)

        Returns:
            Tuple of (global_ids, sequences, length) or None
        """
        min_size = min_size if min_size is not None else self.min_group_size
        max_size = max_size if max_size is not None else self.max_group_size

        # Find the largest group that meets size requirements
        best_length = None
        best_group = None
        best_size = 0

        for length, group in self._length_groups.items():
            group_size = len(group)
            if group_size >= min_size and group_size > best_size:
                if max_size is None or group_size <= max_size:
                    best_length = length
                    best_group = group
                    best_size = group_size
                elif best_size == 0:
                    # Take first group even if too large (will be split)
                    best_length = length
                    best_group = group
                    best_size = group_size

        if best_group is None:
            return None

        # Extract up to max_size sequences
        if max_size is not None and len(best_group) > max_size:
            best_group = best_group[:max_size]

        global_ids = [gid for gid, _ in best_group]
        sequences = [seq for _, seq in best_group]

        # Remove from pool (will be re-added after processing if still active)
        for gid, _ in best_group:
            length, is_active = self._sequence_metadata.get(gid, (None, False))
            if length is not None:
                self._length_groups[length] = [
                    (gid2, seq2)
                    for gid2, seq2 in self._length_groups[length]
                    if gid2 != gid
                ]
                if not self._length_groups[length]:
                    del self._length_groups[length]

        # Update statistics
        self._stats["total_groups_formed"] += 1.0
        tokens_in_group = float(best_length * len(global_ids))
        self._stats["same_length_tokens"] += tokens_in_group
        self._stats["total_tokens_processed"] += tokens_in_group

        return (global_ids, sequences, best_length)

    def get_mixed_length_batch(
        self,
        max_size: Optional[int] = None,
    ) -> Optional[Tuple[List[int], List[torch.Tensor]]]:
        """
        Get a mixed-length batch (fallback when no same-length groups available).

        Args:
            max_size: Maximum batch size (None = all remaining)

        Returns:
            Tuple of (global_ids, sequences) or None
        """
        # Collect all active sequences
        all_sequences = []
        for length, group in self._length_groups.items():
            for gid, seq in group:
                all_sequences.append((gid, seq))

        if not all_sequences:
            return None

        # Limit batch size
        if max_size is not None and len(all_sequences) > max_size:
            all_sequences = all_sequences[:max_size]

        global_ids = [gid for gid, _ in all_sequences]
        sequences = [seq for _, seq in all_sequences]

        # Remove from pool
        for gid, _ in all_sequences:
            length, _ = self._sequence_metadata.get(gid, (None, False))
            if length is not None:
                self._length_groups[length] = [
                    (gid2, seq2)
                    for gid2, seq2 in self._length_groups[length]
                    if gid2 != gid
                ]
                if not self._length_groups[length]:
                    del self._length_groups[length]

        # Update statistics
        total_tokens = float(sum(seq.shape[0] for _, seq in all_sequences))
        self._stats["mixed_length_tokens"] += total_tokens
        self._stats["total_tokens_processed"] += total_tokens

        return (global_ids, sequences)

    def get_next_batch(
        self,
        prefer_same_length: bool = True,
        min_group_size: Optional[int] = None,
        max_group_size: Optional[int] = None,
    ) -> Tuple[Optional[List[int]], Optional[List[torch.Tensor]], bool]:
        """
        Get next batch, preferring same-length groups.

        Args:
            prefer_same_length: If True, try same-length groups first
            min_group_size: Override min_group_size for this call
            max_group_size: Override max_group_size for this call

        Returns:
            Tuple of (global_ids, sequences, is_same_length)
            Returns (None, None, False) if no sequences available
        """
        if prefer_same_length:
            # Try to get same-length group first
            same_length_group = self.get_same_length_group(
                min_size=min_group_size,
                max_size=max_group_size,
            )
            if same_length_group is not None:
                global_ids, sequences, _ = same_length_group
                return (global_ids, sequences, True)

        # Fall back to mixed-length batch
        mixed_batch = self.get_mixed_length_batch()
        if mixed_batch is not None:
            global_ids, sequences = mixed_batch
            return (global_ids, sequences, False)

        return (None, None, False)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get pool statistics.

        Returns:
            Dictionary with statistics
        """
        stats = dict(self._stats)

        # Calculate percentages
        if stats["total_tokens_processed"] > 0:
            stats["same_length_percentage"] = (
                stats["same_length_tokens"] / stats["total_tokens_processed"] * 100.0
            )
            stats["mixed_length_percentage"] = (
                stats["mixed_length_tokens"] / stats["total_tokens_processed"] * 100.0
            )
        else:
            stats["same_length_percentage"] = 0.0
            stats["mixed_length_percentage"] = 0.0

        # Average group size
        if stats["total_groups_formed"] > 0:
            stats["avg_group_size"] = (
                stats["same_length_tokens"] / stats["total_groups_formed"]
                if stats["total_groups_formed"] > 0
                else 0.0
            )
        else:
            stats["avg_group_size"] = 0.0

        # Current pool state
        stats["current_groups"] = float(len(self._length_groups))
        stats["current_sequences"] = float(len(self._sequence_metadata))

        return stats

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_groups_formed": 0.0,
            "same_length_tokens": 0.0,
            "mixed_length_tokens": 0.0,
            "total_tokens_processed": 0.0,
        }

    def clear(self) -> None:
        """Clear all sequences from the pool."""
        self._length_groups.clear()
        self._sequence_metadata.clear()
        self.reset_statistics()

    def __len__(self) -> int:
        """Return number of active sequences in pool."""
        return len(self._sequence_metadata)
