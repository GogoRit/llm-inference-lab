"""
Centralized KV Cache Manager for Speculative Decoding

Manages KV cache state with proper batch tracking to prevent dimension mismatches.
Uses pre-allocated static buffers to avoid dynamic memory reallocation overhead.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SafeKVCacheManager:
    """
    Centralized KV cache manager with pre-allocated static buffers.

    Uses pre-allocated tensors of shape [batch_size, num_heads, max_seq_len, head_dim]
    to avoid expensive torch.cat() operations. Tracks current sequence length per
    batch position and returns sliced views for compatibility with existing code.

    Tracks KV cache per global sequence ID and ensures alignment with
    actual sequence lengths (excluding padding).
    """

    def __init__(
        self,
        device: str,
        max_seq_len: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the KV cache manager.

        Args:
            device: Device to allocate buffers on
            max_seq_len: Maximum sequence length (lazy-initialized if None)
            num_heads: Number of attention heads (lazy-initialized if None)
            head_dim: Dimension of each attention head (lazy-initialized if None)
            num_layers: Number of transformer layers (lazy-initialized if None)
            dtype: Data type for cache tensors (lazy-initialized if None)
        """
        self.device = device
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype

        # Pre-allocated static buffers: Dict[layer_idx] -> [key_buffer, value_buffer]
        # Shape: [batch_size, num_heads, max_seq_len, head_dim]
        self.base_cache: Optional[Dict[int, List[torch.Tensor]]] = None
        self.draft_cache: Optional[Dict[int, List[torch.Tensor]]] = None

        # Track current sequence length per batch position
        # Maps batch_position -> current_seq_len
        # Note: batch_position is the index in the current batch (0 to batch_size-1)
        self.base_current_seq_lens: List[int] = []
        self.draft_current_seq_lens: List[int] = []

        self.current_batch_size: int = 0
        self.active_indices: Optional[torch.Tensor] = None

        # Track sequence lengths per global sequence ID (for compatibility)
        # Maps global_sequence_id -> actual_sequence_length (excluding padding)
        self.base_sequence_lengths: Dict[int, int] = {}
        self.draft_sequence_lengths: Dict[int, int] = {}

        # Track global sequence IDs for current batch
        # Maps batch_index -> global_sequence_id
        self.batch_to_global_map: Dict[int, int] = {}

        # Flag to track if buffers are initialized
        self._buffers_initialized = False

    def reset(self) -> None:
        """
        Reset cache state while preserving pre-allocated buffers.

        CRITICAL: This acts as a ring buffer reset - we keep the heavy GPU tensors
        (base_cache, draft_cache) alive to prevent VRAM fragmentation and OOM errors
        during consecutive runs in K-sweep benchmarks. Only the sequence length pointers
        are reset to zero, allowing the same VRAM block to be reused.
        """
        # PRESERVE buffers to act as ring buffer (prevent OOM/fragmentation)
        # Do NOT clear self.base_cache or self.draft_cache - keep GPU tensors alive

        # Only reset sequence length pointers (ring buffer pointers)
        # Preserve batch_size from existing state if buffers exist
        if self.base_cache is not None:
            # Reset pointers to 0 for all batch positions, preserving batch size
            batch_size = (
                self.current_batch_size
                if self.current_batch_size > 0
                else (
                    len(self.base_current_seq_lens) if self.base_current_seq_lens else 0
                )
            )
            self.base_current_seq_lens = [0] * batch_size if batch_size > 0 else []
        else:
            self.base_current_seq_lens = []

        if self.draft_cache is not None:
            # Reset pointers to 0 for all batch positions, preserving batch size
            batch_size = (
                self.current_batch_size
                if self.current_batch_size > 0
                else (
                    len(self.draft_current_seq_lens)
                    if self.draft_current_seq_lens
                    else 0
                )
            )
            self.draft_current_seq_lens = [0] * batch_size if batch_size > 0 else []
        else:
            self.draft_current_seq_lens = []

        # Reset metadata tracking (these are lightweight)
        # Note: current_batch_size is preserved if buffers exist (for ring buffer reuse)
        # Only reset if no buffers exist
        if self.base_cache is None and self.draft_cache is None:
            self.current_batch_size = 0
        self.active_indices = None
        self.base_sequence_lengths = {}
        self.draft_sequence_lengths = {}
        self.batch_to_global_map = {}

        # Note: _buffers_initialized flag is preserved - buffers remain initialized

    def set_batch_size(self, batch_size: int) -> None:
        """Set current batch size - resets cache if batch size changes."""
        if self.current_batch_size != batch_size:
            self.reset()
            self.current_batch_size = batch_size
            # Reset sequence length tracking for new batch size
            self.base_current_seq_lens = [0] * batch_size
            self.draft_current_seq_lens = [0] * batch_size

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

    def _ensure_buffers_initialized(
        self,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        cache_type: str = "base",
    ) -> None:
        """
        Ensure pre-allocated buffers are initialized for the given dimensions.

        Args:
            batch_size: Batch size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            dtype: Data type for buffers
            cache_type: "base" or "draft"
        """
        cache = self.base_cache if cache_type == "base" else self.draft_cache
        if cache is not None and len(cache) > 0:
            # Check if existing buffers match dimensions
            first_key = list(cache.values())[0][0]
            if (
                first_key.shape[0] == batch_size
                and first_key.shape[1] == num_heads
                and first_key.shape[2] == max_seq_len
                and first_key.shape[3] == head_dim
                and first_key.dtype == dtype
                and len(cache) == num_layers
            ):
                # Buffers already initialized with correct dimensions
                return

        # Allocate new buffers
        device_obj = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        new_cache = {}
        for layer_idx in range(num_layers):
            key_buffer = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device_obj,
            )
            value_buffer = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device_obj,
            )
            new_cache[layer_idx] = [key_buffer, value_buffer]

        if cache_type == "base":
            self.base_cache = new_cache
            self.base_current_seq_lens = [0] * batch_size
        else:
            self.draft_cache = new_cache
            self.draft_current_seq_lens = [0] * batch_size

        # Update configuration
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self._buffers_initialized = True

    def filter_kv_cache(
        self,
        cache: Optional[Dict[int, List[torch.Tensor]]],
        active_indices: Optional[torch.Tensor] = None,
        current_seq_lens: Optional[List[int]] = None,
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Filter entire KV cache by active indices and return sliced views.

        Args:
            cache: Dict mapping layer_idx -> [key_buffer, value_buffer]
            active_indices: Optional tensor of active indices
            current_seq_lens: Optional list of current sequence lengths per batch position

        Returns:
            Tuple of (key, value) tuples per layer, sliced to current_seq_len, or None if filtering fails
        """
        if cache is None or len(cache) == 0:
            return None

        indices = active_indices if active_indices is not None else self.active_indices
        if indices is None or indices.numel() == 0:
            return None

        # Get current sequence lengths for active indices
        if current_seq_lens is None:
            # Use max length across all sequences (conservative)
            max_len = (
                max(self.base_current_seq_lens) if self.base_current_seq_lens else 0
            )
            seq_lens = [max_len] * indices.numel()
        else:
            # Use provided sequence lengths, indexed by active_indices
            if isinstance(indices, torch.Tensor):
                indices_list = indices.cpu().tolist()
            else:
                indices_list = list(indices)
            seq_lens = [
                current_seq_lens[i] if i < len(current_seq_lens) else 0
                for i in indices_list
            ]

        filtered_layers = []
        for layer_idx in sorted(cache.keys()):
            key_buffer, value_buffer = cache[layer_idx]
            filtered_key = self.filter_cache_layer(key_buffer, indices)
            filtered_value = self.filter_cache_layer(value_buffer, indices)

            if filtered_key is None or filtered_value is None:
                return None  # Filtering failed

            # Slice to current sequence length (return view, not copy)
            # Handle variable lengths by using max length (padding will be handled by attention mask)
            max_seq_len = max(seq_lens) if seq_lens else filtered_key.shape[2]
            filtered_layers.append(
                (
                    filtered_key[:, :, :max_seq_len, :],
                    filtered_value[:, :, :max_seq_len, :],
                )
            )

        return tuple(filtered_layers)

    def update_base_cache(
        self,
        new_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        active_indices: List[int],
        tokens_appended: Optional[List[int]] = None,
    ) -> None:
        """
        Update base cache with new KV cache using in-place assignment to pre-allocated buffers.

        Args:
            new_kv: New KV cache tuple for all active sequences
            active_indices: List of active sequence indices (batch positions)
            tokens_appended: Optional list of number of tokens appended per sequence
        """
        if new_kv is None or len(new_kv) == 0:
            return

        # Infer dimensions from first KV cache if not initialized
        if not self._buffers_initialized or self.base_cache is None:
            first_key, first_value = new_kv[0]
            batch_dim = first_key.shape[0]
            num_heads = first_key.shape[1]
            new_seq_len = first_key.shape[2]
            head_dim = first_key.shape[3]
            num_layers = len(new_kv)
            dtype = first_key.dtype

            # Determine max_seq_len if not set
            max_seq_len = self.max_seq_len
            if max_seq_len is None:
                # Estimate: use current sequence length + generous buffer (e.g., 2x)
                # This is a conservative estimate; caller should set max_seq_len explicitly
                estimated_max = new_seq_len * 4  # 4x buffer for safety
                max_seq_len = estimated_max
                logger.warning(
                    f"max_seq_len not set, estimating {max_seq_len} from first KV cache. "
                    "Consider setting max_seq_len explicitly for better memory efficiency."
                )

            # Ensure batch size is set
            if self.current_batch_size == 0:
                self.current_batch_size = batch_dim
                self.base_current_seq_lens = [0] * batch_dim

            # Initialize buffers
            self._ensure_buffers_initialized(
                batch_size=self.current_batch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                dtype=dtype,
                cache_type="base",
            )

        # Convert active_indices to list if tensor
        if isinstance(active_indices, torch.Tensor):
            active_indices_list = active_indices.cpu().tolist()
        else:
            active_indices_list = list(active_indices)

        # Update each layer with in-place assignment
        for layer_idx, (key, value) in enumerate(new_kv):
            if layer_idx not in self.base_cache:
                raise RuntimeError(
                    f"Layer {layer_idx} not found in pre-allocated cache. "
                    "This should not happen if buffers were initialized correctly."
                )

            key_buffer, value_buffer = self.base_cache[layer_idx]
            new_seq_len = key.shape[2]  # Length of new KV cache

            # Update each active sequence
            for i, batch_pos in enumerate(active_indices_list):
                if batch_pos >= len(self.base_current_seq_lens):
                    logger.warning(
                        f"batch_pos {batch_pos} >= len(current_seq_lens) {len(self.base_current_seq_lens)}, skipping"
                    )
                    continue

                current_pos = self.base_current_seq_lens[batch_pos]
                new_pos = current_pos + new_seq_len

                # Safety check: ensure we don't exceed max_seq_len
                if new_pos > self.max_seq_len:
                    raise RuntimeError(
                        f"KV cache overflow: attempting to write {new_pos} tokens "
                        f"but max_seq_len is {self.max_seq_len}. "
                        f"batch_pos={batch_pos}, current_pos={current_pos}, new_seq_len={new_seq_len}"
                    )

                # In-place assignment: write new KV cache at current position
                key_buffer[batch_pos, :, current_pos:new_pos, :] = (
                    key[i].detach().contiguous()
                )
                value_buffer[batch_pos, :, current_pos:new_pos, :] = (
                    value[i].detach().contiguous()
                )

                # Update current sequence length
                self.base_current_seq_lens[batch_pos] = new_pos

        # Update sequence lengths if tokens_appended is provided (for compatibility)
        if tokens_appended is not None and len(tokens_appended) == len(
            active_indices_list
        ):
            for i, (batch_idx, tokens_added) in enumerate(
                zip(active_indices_list, tokens_appended)
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
        """
        Update draft cache with new tokens using in-place assignment to pre-allocated buffers.

        Args:
            new_kv: New KV cache tuple for all active sequences
        """
        if new_kv is None or len(new_kv) == 0:
            return

        # Infer dimensions from first KV cache if not initialized
        if self.draft_cache is None or len(self.draft_cache) == 0:
            first_key, first_value = new_kv[0]
            batch_dim = first_key.shape[0]
            num_heads = first_key.shape[1]
            new_seq_len = first_key.shape[2]
            head_dim = first_key.shape[3]
            num_layers = len(new_kv)
            dtype = first_key.dtype

            # Determine max_seq_len if not set (use same as base cache if available)
            max_seq_len = self.max_seq_len
            if max_seq_len is None:
                estimated_max = new_seq_len * 4
                max_seq_len = estimated_max
                logger.warning(
                    f"max_seq_len not set for draft cache, estimating {max_seq_len}. "
                    "Consider setting max_seq_len explicitly."
                )

            # Ensure batch size is set
            if self.current_batch_size == 0:
                self.current_batch_size = batch_dim
                self.draft_current_seq_lens = [0] * batch_dim

            # Initialize buffers
            self._ensure_buffers_initialized(
                batch_size=self.current_batch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                dtype=dtype,
                cache_type="draft",
            )

        # Update each layer with in-place assignment
        for layer_idx, (key, value) in enumerate(new_kv):
            if layer_idx not in self.draft_cache:
                raise RuntimeError(
                    f"Layer {layer_idx} not found in pre-allocated draft cache."
                )

            key_buffer, value_buffer = self.draft_cache[layer_idx]
            new_seq_len = key.shape[2]
            batch_dim = key.shape[0]

            # Update each sequence in batch
            for batch_pos in range(batch_dim):
                if batch_pos >= len(self.draft_current_seq_lens):
                    logger.warning(
                        f"batch_pos {batch_pos} >= len(draft_current_seq_lens), skipping"
                    )
                    continue

                current_pos = self.draft_current_seq_lens[batch_pos]
                new_pos = current_pos + new_seq_len

                # Safety check
                if new_pos > self.max_seq_len:
                    raise RuntimeError(
                        f"Draft KV cache overflow: attempting to write {new_pos} tokens "
                        f"but max_seq_len is {self.max_seq_len}. "
                        f"batch_pos={batch_pos}, current_pos={current_pos}, new_seq_len={new_seq_len}"
                    )

                # In-place assignment
                key_buffer[batch_pos, :, current_pos:new_pos, :] = (
                    key[batch_pos].detach().contiguous()
                )
                value_buffer[batch_pos, :, current_pos:new_pos, :] = (
                    value[batch_pos].detach().contiguous()
                )

                # Update current sequence length
                self.draft_current_seq_lens[batch_pos] = new_pos

    def get_base_past_kv(
        self, active_indices: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Get filtered base past_key_values for active sequences.

        Returns sliced views of pre-allocated buffers, sliced to current sequence length.
        """
        return self.filter_kv_cache(
            self.base_cache, active_indices, self.base_current_seq_lens
        )

    def get_draft_past_kv(
        self, active_indices: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Get filtered draft past_key_values for active sequences.

        Returns sliced views of pre-allocated buffers, sliced to current sequence length.
        """
        return self.filter_kv_cache(
            self.draft_cache, active_indices, self.draft_current_seq_lens
        )

    def get_current_cache(
        self,
        cache_type: str = "base",
        active_indices: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        """
        Get current cache as sliced views of pre-allocated buffers.

        This is the main accessor method that returns views sliced to current sequence length.
        Compatible with PyTorch attention operations.

        Args:
            cache_type: "base" or "draft"
            active_indices: Optional tensor of active indices

        Returns:
            Tuple of (key, value) tuples per layer, sliced to current_seq_len
        """
        cache = self.base_cache if cache_type == "base" else self.draft_cache
        current_seq_lens = (
            self.base_current_seq_lens
            if cache_type == "base"
            else self.draft_current_seq_lens
        )
        return self.filter_kv_cache(cache, active_indices, current_seq_lens)

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

        With pre-allocated buffers, this method updates the current_seq_lens tracking
        to reflect the new sequence lengths. The actual KV data in buffers remains unchanged
        (views will automatically reflect the new lengths).

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
        current_seq_lens = (
            self.base_current_seq_lens
            if cache_type == "base"
            else self.draft_current_seq_lens
        )

        if cache is None or len(cache) == 0:
            # No cache to realign, just update lengths
            for global_id, new_length in zip(global_sequence_ids, new_sequence_lengths):
                lengths_dict[global_id] = new_length
            return

        # Update current_seq_lens for each batch position
        # Assumption: global_sequence_ids[i] corresponds to batch position i
        for batch_idx, (global_id, new_length) in enumerate(
            zip(global_sequence_ids, new_sequence_lengths)
        ):
            # Clamp new_length to max_seq_len (safety check)
            if self.max_seq_len is not None and new_length > self.max_seq_len:
                logger.warning(
                    f"new_length {new_length} > max_seq_len {self.max_seq_len}, clamping"
                )
                new_length = self.max_seq_len

            # Update tracking
            if batch_idx < len(current_seq_lens):
                current_seq_lens[batch_idx] = new_length
            lengths_dict[global_id] = new_length

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
