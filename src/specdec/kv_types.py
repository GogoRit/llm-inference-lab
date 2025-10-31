"""
KV Cache Types for Speculative Decoding

Defines data structures for managing key-value caches in transformer models,
enabling efficient token appending without recomputation.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class KVCache:
    """
    Container for transformer key-value cache.

    Attributes:
        past_key_values: Tuple of (key, value) tensors per layer.
                        Each key/value is typically
        [batch, num_heads, seq_len, head_dim]
        seq_len: Current sequence length in the cache
        dtype: Data type of the cache tensors
        device: Device where cache tensors are stored
    """

    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    seq_len: int
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_hf_output(
        cls, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    ) -> "KVCache":
        """
        Create KVCache from HuggingFace model output.

        Args:
            past_key_values: past_key_values from model output

        Returns:
            KVCache instance
        """
        if not past_key_values or len(past_key_values) == 0:
            raise ValueError("Cannot create KVCache from empty past_key_values")

        # Get metadata from first layer's key tensor
        first_key = past_key_values[0][0]
        seq_len = first_key.shape[2]  # [batch, num_heads, seq_len, head_dim]
        dtype = first_key.dtype
        device = first_key.device

        return cls(
            past_key_values=past_key_values,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
        )

    def slice_prefix(self, length: int) -> "KVCache":
        """
        Extract a prefix of the cache.

        Args:
            length: Number of positions to extract

        Returns:
            New KVCache with sliced tensors
        """
        if length > self.seq_len:
            raise ValueError(
                f"Cannot slice length {length} from cache with seq_len {self.seq_len}"
            )

        sliced_kv = tuple(
            (k[:, :, :length, :], v[:, :, :length, :]) for k, v in self.past_key_values
        )

        return KVCache(
            past_key_values=sliced_kv,
            seq_len=length,
            dtype=self.dtype,
            device=self.device,
        )

    def to(self, device: torch.device) -> "KVCache":
        """
        Move cache to a different device.

        Args:
            device: Target device

        Returns:
            New KVCache on target device
        """
        if self.device == device:
            return self

        moved_kv = tuple((k.to(device), v.to(device)) for k, v in self.past_key_values)

        return KVCache(
            past_key_values=moved_kv,
            seq_len=self.seq_len,
            dtype=self.dtype,
            device=device,
        )

    def get_num_layers(self) -> int:
        """Get number of transformer layers in cache."""
        return len(self.past_key_values)

    def get_shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Get shapes of key and value tensors (from first layer)."""
        if len(self.past_key_values) == 0:
            return ((), ())
        k, v = self.past_key_values[0]
        return (tuple(k.shape), tuple(v.shape))


def validate_kv_compatibility(base_cache: KVCache, new_cache: KVCache) -> None:
    """
    Validate that two caches are compatible for appending.

    Args:
        base_cache: Existing base cache
        new_cache: New cache to append

    Raises:
        ValueError: If caches are incompatible
    """
    if base_cache.get_num_layers() != new_cache.get_num_layers():
        raise ValueError(
            f"Layer count mismatch: base={base_cache.get_num_layers()}, "
            f"new={new_cache.get_num_layers()}"
        )

    if base_cache.dtype != new_cache.dtype:
        raise ValueError(
            f"Dtype mismatch: base={base_cache.dtype}, new={new_cache.dtype}"
        )

    # Check shapes (excluding seq_len dimension)
    base_shapes = base_cache.get_shapes()
    new_shapes = new_cache.get_shapes()

    # Check batch, num_heads, head_dim match
    if (
        base_shapes[0][0] != new_shapes[0][0]
        or base_shapes[0][1] != new_shapes[0][1]
        or base_shapes[0][3] != new_shapes[0][3]
    ):
        raise ValueError(
            f"Shape mismatch (excluding seq_len): base={base_shapes}, new={new_shapes}"
        )
