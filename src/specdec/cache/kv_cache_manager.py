"""
Centralized KV Cache Manager for Speculative Decoding

Manages KV cache state with proper batch tracking to prevent dimension mismatches.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SafeKVCacheManager:
    """
    Centralized KV cache manager with explicit sequence length tracking.

    Tracks KV cache per global sequence ID and ensures alignment with
    actual sequence lengths (excluding padding).
    """

    def __init__(self, device: str):
        """Initialize the KV cache manager."""
        self.device = device
        self.base_cache: Optional[Dict[int, List[torch.Tensor]]] = None
        self.draft_cache: Optional[Dict[int, List[torch.Tensor]]] = None
        self.current_batch_size: int = 0
        self.active_indices: Optional[torch.Tensor] = None

        # Track sequence lengths per global sequence ID
        # Maps global_sequence_id -> actual_sequence_length (excluding padding)
        self.base_sequence_lengths: Dict[int, int] = {}
        self.draft_sequence_lengths: Dict[int, int] = {}

        # Track global sequence IDs for current batch
        # Maps batch_index -> global_sequence_id
        self.batch_to_global_map: Dict[int, int] = {}

    def reset(self) -> None:
        """Reset all cache state."""
        self.base_cache = None
        self.draft_cache = None
        self.current_batch_size = 0
        self.active_indices = None
        self.base_sequence_lengths = {}
        self.draft_sequence_lengths = {}
        self.batch_to_global_map = {}

    def set_batch_size(self, batch_size: int) -> None:
        """Set current batch size - resets cache if batch size changes."""
        if self.current_batch_size != batch_size:
            self.reset()
            self.current_batch_size = batch_size

    def set_active_indices(self, active_indices: List[int]) -> None:
        """Set active indices for current batch."""
        if isinstance(active_indices, torch.Tensor):
            self.active_indices = active_indices
        else:
            # Convert device string to torch.device if needed
            device_obj = (
                torch.device(self.device)
                if isinstance(self.device, str)
                else self.device
            )
            self.active_indices = torch.tensor(
                active_indices, dtype=torch.long, device=device_obj
            )

    def filter_cache_layer(
        self,
        cache_tensor: torch.Tensor,
        active_indices: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Efficiently filter cache tensor by active indices.

        Args:
            cache_tensor: KV cache tensor [batch, ...]
            active_indices: Optional tensor of active indices (uses self.active_indices if None)

        Returns:
            Filtered cache tensor or None if no active indices
        """
        if cache_tensor is None:
            return None

        indices = active_indices if active_indices is not None else self.active_indices
        if indices is None or indices.numel() == 0:
            return None

        batch_size = cache_tensor.shape[0]

        # Fast path: all indices active
        if indices.numel() == batch_size and torch.all(
            indices == torch.arange(batch_size, device=self.device)
        ):
            return cache_tensor

        # Validate indices are in range
        max_idx = indices.max().item() if indices.numel() > 0 else -1
        if max_idx >= batch_size:
            # Invalid indices - return None rather than crashing
            logger.warning(
                f"KV cache: max_index {max_idx} >= batch_size {batch_size}, returning None"
            )
            return None

        # Efficient filtering
        return cache_tensor.index_select(0, indices)

    def filter_kv_cache(
        self,
        cache: Optional[Dict[int, List[torch.Tensor]]],
        active_indices: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Filter entire KV cache by active indices.

        Args:
            cache: Dict mapping layer_idx -> [key_tensor, value_tensor]
            active_indices: Optional tensor of active indices

        Returns:
            Tuple of (key, value) tuples per layer, or None if filtering fails
        """
        if cache is None or len(cache) == 0:
            return None

        indices = active_indices if active_indices is not None else self.active_indices
        if indices is None or indices.numel() == 0:
            return None

        filtered_layers = []
        for layer_idx in sorted(cache.keys()):
            key_tensor, value_tensor = cache[layer_idx]
            filtered_key = self.filter_cache_layer(key_tensor, indices)
            filtered_value = self.filter_cache_layer(value_tensor, indices)

            if filtered_key is None or filtered_value is None:
                return None  # Filtering failed

            filtered_layers.append((filtered_key, filtered_value))

        return tuple(filtered_layers)

    def update_base_cache(
        self,
        new_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        active_indices: List[int],
        tokens_appended: Optional[List[int]] = None,
    ) -> None:
        """
        Update base cache with new KV cache for active sequences.

        Args:
            new_kv: New KV cache tuple for all active sequences
            active_indices: List of active sequence indices (for filtering if needed)
            tokens_appended: Optional list of number of tokens appended per sequence
        """
        if new_kv is None or len(new_kv) == 0:
            return

        # Initialize cache if needed
        if self.base_cache is None:
            self.base_cache = {}

        # Convert active_indices to tensor if needed
        if isinstance(active_indices, torch.Tensor):
            active_tensor = active_indices
        else:
            if len(new_kv) > 0:
                device = new_kv[0][0].device
                active_tensor = torch.tensor(
                    active_indices, dtype=torch.long, device=device
                )
            else:
                return

        for layer_idx, (key, value) in enumerate(new_kv):
            # key/value shape: [active_count, num_heads, seq_len, head_dim]
            batch_dim = key.shape[0]

            # Append to existing cache or create new
            if layer_idx in self.base_cache:
                existing_key = self.base_cache[layer_idx][0]
                existing_value = self.base_cache[layer_idx][1]

                # Check if existing cache batch dimension matches
                if existing_key.shape[0] == batch_dim:
                    # Direct append along sequence dimension (dim=2)
                    self.base_cache[layer_idx][0] = torch.cat(
                        [existing_key, key.detach().contiguous()], dim=2
                    )
                    self.base_cache[layer_idx][1] = torch.cat(
                        [existing_value, value.detach().contiguous()], dim=2
                    )
                elif existing_key.shape[
                    0
                ] == self.current_batch_size and batch_dim == len(active_indices):
                    # Filter existing cache to active indices, then append
                    filtered_key = self.filter_cache_layer(existing_key, active_tensor)
                    filtered_value = self.filter_cache_layer(
                        existing_value, active_tensor
                    )

                    if filtered_key is not None and filtered_value is not None:
                        self.base_cache[layer_idx][0] = torch.cat(
                            [filtered_key, key.detach().contiguous()], dim=2
                        )
                        self.base_cache[layer_idx][1] = torch.cat(
                            [filtered_value, value.detach().contiguous()], dim=2
                        )
                    else:
                        # Filtering failed - reset with new cache
                        self.base_cache[layer_idx] = [
                            key.detach().contiguous(),
                            value.detach().contiguous(),
                        ]
                else:
                    # Batch dimension mismatch - reset
                    self.base_cache[layer_idx] = [
                        key.detach().contiguous(),
                        value.detach().contiguous(),
                    ]
            else:
                # New layer - initialize
                self.base_cache[layer_idx] = [
                    key.detach().contiguous(),
                    value.detach().contiguous(),
                ]

        # Update sequence lengths if tokens_appended is provided
        if tokens_appended is not None and len(tokens_appended) == len(active_indices):
            # Get global sequence IDs for active indices
            for i, (batch_idx, tokens_added) in enumerate(
                zip(active_indices, tokens_appended)
            ):
                global_id = self.batch_to_global_map.get(batch_idx, None)
                if global_id is not None:
                    current_length = self.base_sequence_lengths.get(global_id, 0)
                    self.base_sequence_lengths[global_id] = (
                        current_length + tokens_added
                    )

    def update_draft_cache(
        self,
        new_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    ) -> None:
        """Update draft cache with new tokens."""
        if new_kv is None or len(new_kv) == 0:
            return

        # Initialize cache if needed
        if self.draft_cache is None:
            self.draft_cache = {}

        for layer_idx, (key, value) in enumerate(new_kv):
            if layer_idx in self.draft_cache:
                existing_key = self.draft_cache[layer_idx][0]
                existing_value = self.draft_cache[layer_idx][1]

                # Verify batch dimension matches
                if existing_key.shape[0] == key.shape[0]:
                    # Append along sequence dimension (dim=2 for [B, H, L, D])
                    self.draft_cache[layer_idx][0] = torch.cat(
                        [existing_key, key.detach().contiguous()], dim=2
                    )
                    self.draft_cache[layer_idx][1] = torch.cat(
                        [existing_value, value.detach().contiguous()], dim=2
                    )
                else:
                    # Batch mismatch - reset
                    self.draft_cache[layer_idx] = [
                        key.detach().contiguous(),
                        value.detach().contiguous(),
                    ]
            else:
                # New layer - initialize
                self.draft_cache[layer_idx] = [
                    key.detach().contiguous(),
                    value.detach().contiguous(),
                ]

    def get_base_past_kv(
        self, active_indices: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """Get filtered base past_key_values for active sequences."""
        return self.filter_kv_cache(self.base_cache, active_indices)

    def get_draft_past_kv(
        self, active_indices: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """Get filtered draft past_key_values for active sequences."""
        return self.filter_kv_cache(self.draft_cache, active_indices)

    def set_sequence_metadata(
        self,
        global_sequence_ids: List[int],
        sequence_lengths: List[int],
        batch_to_global_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Set sequence metadata for current batch.

        Args:
            global_sequence_ids: List of global sequence IDs for active sequences
            sequence_lengths: List of actual sequence lengths (excluding padding) for each active sequence
            batch_to_global_map: Optional mapping from batch index to global sequence ID
        """
        if len(global_sequence_ids) != len(sequence_lengths):
            raise ValueError(
                f"global_sequence_ids ({len(global_sequence_ids)}) and "
                f"sequence_lengths ({len(sequence_lengths)}) must have same length"
            )

        # Update sequence lengths for base cache
        for global_id, length in zip(global_sequence_ids, sequence_lengths):
            self.base_sequence_lengths[global_id] = length
            self.draft_sequence_lengths[global_id] = length

        # Update batch to global mapping
        if batch_to_global_map is not None:
            self.batch_to_global_map = batch_to_global_map
        else:
            # Create default mapping: batch_idx -> global_id
            self.batch_to_global_map = {
                i: global_id for i, global_id in enumerate(global_sequence_ids)
            }

    def get_sequence_lengths(
        self, global_sequence_ids: List[int], cache_type: str = "base"
    ) -> List[int]:
        """
        Get current sequence lengths for specified global sequence IDs.

        Args:
            global_sequence_ids: List of global sequence IDs
            cache_type: "base" or "draft"

        Returns:
            List of sequence lengths
        """
        lengths_dict = (
            self.base_sequence_lengths
            if cache_type == "base"
            else self.draft_sequence_lengths
        )
        return [lengths_dict.get(global_id, 0) for global_id in global_sequence_ids]

    def realign_kv_cache(
        self,
        global_sequence_ids: List[int],
        new_sequence_lengths: List[int],
        cache_type: str = "base",
    ) -> None:
        """
        Realign KV cache to match new sequence lengths after unpadding/repadding.

        This compacts KV cache to remove padding entries and ensure each sequence's
        KV cache matches its actual token length. The global_sequence_ids list should
        be in the same order as the current batch (i.e., batch_idx i corresponds to
        global_sequence_ids[i]).

        Args:
            global_sequence_ids: List of global sequence IDs in current batch order
            new_sequence_lengths: New actual sequence lengths (excluding padding) for each sequence
            cache_type: "base" or "draft"
        """
        if len(global_sequence_ids) != len(new_sequence_lengths):
            raise ValueError(
                f"global_sequence_ids ({len(global_sequence_ids)}) and "
                f"new_sequence_lengths ({len(new_sequence_lengths)}) must have same length"
            )

        cache = self.base_cache if cache_type == "base" else self.draft_cache
        lengths_dict = (
            self.base_sequence_lengths
            if cache_type == "base"
            else self.draft_sequence_lengths
        )

        if cache is None or len(cache) == 0:
            # No cache to realign, just update lengths
            for global_id, new_length in zip(global_sequence_ids, new_sequence_lengths):
                lengths_dict[global_id] = new_length
            return

        # Realign each layer
        # Assumption: global_sequence_ids[i] corresponds to batch position i in current cache
        for layer_idx in sorted(cache.keys()):
            key_tensor, value_tensor = cache[layer_idx]
            # Shape: [batch_size, num_heads, seq_len, head_dim]

            batch_size, num_heads, current_seq_len, head_dim = key_tensor.shape

            if batch_size != len(global_sequence_ids):
                logger.warning(
                    f"Batch size mismatch: cache has {batch_size}, "
                    f"but {len(global_sequence_ids)} sequences provided. "
                    f"Skipping realignment for layer {layer_idx}."
                )
                continue

            # For each sequence, extract KV up to its new length
            realigned_keys = []
            realigned_values = []

            for batch_idx, (global_id, new_length) in enumerate(
                zip(global_sequence_ids, new_sequence_lengths)
            ):
                # Clamp new_length to current_seq_len (can't extract more than exists)
                extract_length = min(new_length, current_seq_len)

                # Extract KV for this sequence up to extract_length
                # Shape: [num_heads, extract_length, head_dim]
                seq_key = key_tensor[batch_idx, :, :extract_length, :].clone()
                seq_value = value_tensor[batch_idx, :, :extract_length, :].clone()

                # If new_length > current_seq_len, we need to pad (shouldn't happen normally)
                if new_length > current_seq_len:
                    pad_length = new_length - current_seq_len
                    seq_key = F.pad(
                        seq_key, (0, 0, 0, pad_length), mode="constant", value=0.0
                    )
                    seq_value = F.pad(
                        seq_value, (0, 0, 0, pad_length), mode="constant", value=0.0
                    )

                realigned_keys.append(seq_key)
                realigned_values.append(seq_value)

                # Update length tracking
                lengths_dict[global_id] = new_length

            if realigned_keys:
                # Stack realigned KV: [batch_size, num_heads, max_new_length, head_dim]
                max_new_length = max(new_sequence_lengths)

                # Pad to max length if needed
                padded_keys = []
                padded_values = []
                for key, value, new_length in zip(
                    realigned_keys, realigned_values, new_sequence_lengths
                ):
                    if new_length < max_new_length:
                        pad_length = max_new_length - new_length
                        # Pad along sequence dimension (dim=1): (left, right) for 1D, but we need (0, 0, 0, pad_length) for 3D
                        # F.pad for 3D: (pad_left_dim2, pad_right_dim2, pad_left_dim1, pad_right_dim1, pad_left_dim0, pad_right_dim0)
                        # We want to pad only the sequence dimension (dim=1), so: (0, 0, 0, pad_length, 0, 0)
                        key_padded = F.pad(
                            key, (0, 0, 0, pad_length, 0, 0), mode="constant", value=0.0
                        )
                        value_padded = F.pad(
                            value,
                            (0, 0, 0, pad_length, 0, 0),
                            mode="constant",
                            value=0.0,
                        )
                        padded_keys.append(key_padded)
                        padded_values.append(value_padded)
                    else:
                        padded_keys.append(key)
                        padded_values.append(value)

                # Stack: [batch_size, num_heads, max_new_length, head_dim]
                new_key = torch.stack(padded_keys, dim=0)
                new_value = torch.stack(padded_values, dim=0)

                # Update cache for this layer
                cache[layer_idx] = [
                    new_key.detach().contiguous(),
                    new_value.detach().contiguous(),
                ]

        # Update batch size
        self.current_batch_size = len(global_sequence_ids)

    def get_kv_shapes(self, cache_type: str = "base") -> Dict[str, Any]:
        """
        Get KV cache shapes for debugging.

        Args:
            cache_type: "base" or "draft"

        Returns:
            Dictionary with cache shapes and metadata
        """
        cache = self.base_cache if cache_type == "base" else self.draft_cache
        lengths_dict = (
            self.base_sequence_lengths
            if cache_type == "base"
            else self.draft_sequence_lengths
        )

        if cache is None or len(cache) == 0:
            return {
                "cache_exists": False,
                "num_layers": 0,
                "sequence_lengths": dict(lengths_dict),
            }

        shapes = {}
        for layer_idx in sorted(cache.keys()):
            key_tensor, value_tensor = cache[layer_idx]
            shapes[f"layer_{layer_idx}"] = {
                "key_shape": list(key_tensor.shape),
                "value_shape": list(value_tensor.shape),
            }

        return {
            "cache_exists": True,
            "num_layers": len(cache),
            "shapes": shapes,
            "sequence_lengths": dict(lengths_dict),
            "batch_size": self.current_batch_size,
        }
