"""
Centralized KV Cache Manager for Speculative Decoding

Manages KV cache state with proper batch tracking to prevent dimension mismatches.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SafeKVCacheManager:
    """
    Centralized KV cache manager with batch dimension tracking.

    Prevents cache reuse issues by tracking batch size and active indices.
    """

    def __init__(self, device: str):
        """Initialize the KV cache manager."""
        self.device = device
        self.base_cache: Optional[Dict[int, List[torch.Tensor]]] = None
        self.draft_cache: Optional[Dict[int, List[torch.Tensor]]] = None
        self.current_batch_size: int = 0
        self.active_indices: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset all cache state."""
        self.base_cache = None
        self.draft_cache = None
        self.current_batch_size = 0
        self.active_indices = None

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
    ) -> None:
        """
        Update base cache with new KV cache for active sequences.

        Args:
            new_kv: New KV cache tuple for all active sequences
            active_indices: List of active sequence indices (for filtering if needed)
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
