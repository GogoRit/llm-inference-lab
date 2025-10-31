"""
Unit tests for KV cache functionality.
"""

import pytest
import torch

from src.kernels.reference import kv_append_ref, kv_append_with_mask_ref
from src.specdec.kv_types import KVCache, validate_kv_compatibility


class TestKVAppendRef:
    """Test PyTorch reference implementation of kv_append."""

    def test_basic_append(self):
        """Test basic KV cache append operation."""
        # Create fake KV cache: 2 layers, batch=1, heads=2, seq=3, dim=4
        B, H, L, D = 1, 2, 3, 4
        K = 2  # New tokens to append

        # Base cache
        base_k = torch.randn(B, H, L, D)
        base_v = torch.randn(B, H, L, D)

        # New cache to append
        new_k = torch.randn(B, H, K, D)
        new_v = torch.randn(B, H, K, D)

        # Append
        out_k, out_v = kv_append_ref(base_k, base_v, new_k, new_v)

        # Check shapes
        assert out_k.shape == (B, H, L + K, D)
        assert out_v.shape == (B, H, L + K, D)

        # Check content: base prefix should match
        assert torch.equal(out_k[:, :, :L, :], base_k)
        assert torch.equal(out_v[:, :, :L, :], base_v)

        # Check content: new suffix should match
        assert torch.equal(out_k[:, :, L:, :], new_k)
        assert torch.equal(out_v[:, :, L:, :], new_v)

    def test_append_single_token(self):
        """Test appending a single token."""
        B, H, L, D = 1, 2, 5, 8
        K = 1

        base_k = torch.ones(B, H, L, D)
        base_v = torch.ones(B, H, L, D) * 2

        new_k = torch.ones(B, H, K, D) * 3
        new_v = torch.ones(B, H, K, D) * 4

        out_k, out_v = kv_append_ref(base_k, base_v, new_k, new_v)

        assert out_k.shape == (B, H, L + K, D)
        assert out_v.shape == (B, H, L + K, D)

        # Check values
        assert torch.allclose(out_k[:, :, :L, :], torch.ones(B, H, L, D))
        assert torch.allclose(out_v[:, :, :L, :], torch.ones(B, H, L, D) * 2)
        assert torch.allclose(out_k[:, :, L:, :], torch.ones(B, H, K, D) * 3)
        assert torch.allclose(out_v[:, :, L:, :], torch.ones(B, H, K, D) * 4)

    def test_append_multiple_heads(self):
        """Test append with multiple attention heads."""
        B, H, L, D = 1, 8, 10, 16
        K = 5

        base_k = torch.randn(B, H, L, D)
        base_v = torch.randn(B, H, L, D)

        new_k = torch.randn(B, H, K, D)
        new_v = torch.randn(B, H, K, D)

        out_k, out_v = kv_append_ref(base_k, base_v, new_k, new_v)

        # Verify each head independently
        for h in range(H):
            assert torch.equal(out_k[:, h, :L, :], base_k[:, h, :, :])
            assert torch.equal(out_v[:, h, :L, :], base_v[:, h, :, :])
            assert torch.equal(out_k[:, h, L:, :], new_k[:, h, :, :])
            assert torch.equal(out_v[:, h, L:, :], new_v[:, h, :, :])

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise assertions."""
        # Mismatched batch size
        base_k = torch.randn(2, 2, 3, 4)
        base_v = torch.randn(2, 2, 3, 4)
        new_k = torch.randn(1, 2, 2, 4)  # Different batch
        new_v = torch.randn(1, 2, 2, 4)

        with pytest.raises(AssertionError):
            kv_append_ref(base_k, base_v, new_k, new_v)

        # Mismatched num heads
        base_k = torch.randn(1, 4, 3, 4)
        base_v = torch.randn(1, 4, 3, 4)
        new_k = torch.randn(1, 2, 2, 4)  # Different heads
        new_v = torch.randn(1, 2, 2, 4)

        with pytest.raises(AssertionError):
            kv_append_ref(base_k, base_v, new_k, new_v)

        # Mismatched head dim
        base_k = torch.randn(1, 2, 3, 8)
        base_v = torch.randn(1, 2, 3, 8)
        new_k = torch.randn(1, 2, 2, 4)  # Different head dim
        new_v = torch.randn(1, 2, 2, 4)

        with pytest.raises(AssertionError):
            kv_append_ref(base_k, base_v, new_k, new_v)


class TestKVAppendWithMask:
    """Test masked KV append (selective appending based on acceptance)."""

    def test_full_acceptance(self):
        """Test when all tokens are accepted."""
        B, H, L, D = 1, 2, 3, 4
        K = 2

        base_k = torch.randn(B, H, L, D)
        base_v = torch.randn(B, H, L, D)
        draft_k = torch.randn(B, H, K, D)
        draft_v = torch.randn(B, H, K, D)

        # All accepted
        accepted_mask = torch.ones(B, K, dtype=torch.uint8)
        accept_len = torch.tensor([K], dtype=torch.int32)

        out_k, out_v = kv_append_with_mask_ref(
            base_k, base_v, draft_k, draft_v, accepted_mask, accept_len
        )

        # Should match simple append
        assert out_k.shape == (B, H, L + K, D)
        assert torch.equal(out_k[:, :, :L, :], base_k)
        assert torch.equal(out_k[:, :, L : L + K, :], draft_k)

    def test_partial_acceptance(self):
        """Test when only some tokens are accepted."""
        B, H, L, D = 1, 2, 3, 4
        K = 4

        base_k = torch.zeros(B, H, L, D)
        base_v = torch.zeros(B, H, L, D)
        draft_k = torch.arange(K).view(1, 1, K, 1).expand(B, H, K, D).float()
        draft_v = torch.arange(K).view(1, 1, K, 1).expand(B, H, K, D).float() * 10

        # Accept first 2 tokens only
        accepted_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.uint8)
        accept_len = torch.tensor([2], dtype=torch.int32)

        out_k, out_v = kv_append_with_mask_ref(
            base_k, base_v, draft_k, draft_v, accepted_mask, accept_len
        )

        # Check that only first 2 positions were appended
        assert torch.allclose(out_k[:, :, L, :], draft_k[:, :, 0, :])
        assert torch.allclose(out_k[:, :, L + 1, :], draft_k[:, :, 1, :])

    def test_zero_acceptance(self):
        """Test when no tokens are accepted."""
        B, H, L, D = 1, 2, 3, 4
        K = 2

        base_k = torch.randn(B, H, L, D)
        base_v = torch.randn(B, H, L, D)
        draft_k = torch.randn(B, H, K, D)
        draft_v = torch.randn(B, H, K, D)

        # None accepted
        accepted_mask = torch.zeros(B, K, dtype=torch.uint8)
        accept_len = torch.tensor([0], dtype=torch.int32)

        out_k, out_v = kv_append_with_mask_ref(
            base_k, base_v, draft_k, draft_v, accepted_mask, accept_len
        )

        # Base cache should be unchanged (but padded)
        assert out_k.shape == (B, H, L + K, D)
        assert torch.equal(out_k[:, :, :L, :], base_k)
        # Rest should be zeros (not filled)
        assert torch.allclose(out_k[:, :, L:, :], torch.zeros(B, H, K, D))


class TestKVCache:
    """Test KVCache dataclass functionality."""

    def test_from_hf_output(self):
        """Test creating KVCache from HF output."""
        # Simulate HF output: tuple of (key, value) per layer
        num_layers = 3
        B, H, L, D = 1, 4, 10, 16

        past_key_values = tuple(
            (torch.randn(B, H, L, D), torch.randn(B, H, L, D))
            for _ in range(num_layers)
        )

        cache = KVCache.from_hf_output(past_key_values)

        assert cache.seq_len == L
        assert cache.get_num_layers() == num_layers
        assert cache.dtype == torch.float32
        assert len(cache.past_key_values) == num_layers

    def test_slice_prefix(self):
        """Test slicing KV cache to a prefix."""
        num_layers = 2
        B, H, L, D = 1, 2, 10, 8

        past_key_values = tuple(
            (torch.randn(B, H, L, D), torch.randn(B, H, L, D))
            for _ in range(num_layers)
        )

        cache = KVCache.from_hf_output(past_key_values)

        # Slice to length 5
        sliced = cache.slice_prefix(5)

        assert sliced.seq_len == 5
        assert sliced.get_num_layers() == num_layers

        # Check that data matches prefix
        for layer_idx in range(num_layers):
            orig_k, orig_v = cache.past_key_values[layer_idx]
            sliced_k, sliced_v = sliced.past_key_values[layer_idx]

            assert sliced_k.shape == (B, H, 5, D)
            assert sliced_v.shape == (B, H, 5, D)
            assert torch.equal(sliced_k, orig_k[:, :, :5, :])
            assert torch.equal(sliced_v, orig_v[:, :, :5, :])

    def test_slice_invalid_length(self):
        """Test that slicing beyond seq_len raises error."""
        num_layers = 2
        B, H, L, D = 1, 2, 10, 8

        past_key_values = tuple(
            (torch.randn(B, H, L, D), torch.randn(B, H, L, D))
            for _ in range(num_layers)
        )

        cache = KVCache.from_hf_output(past_key_values)

        with pytest.raises(ValueError):
            cache.slice_prefix(15)  # Too long

    def test_device_transfer(self):
        """Test moving cache to different device."""
        num_layers = 2
        B, H, L, D = 1, 2, 5, 4

        past_key_values = tuple(
            (torch.randn(B, H, L, D), torch.randn(B, H, L, D))
            for _ in range(num_layers)
        )

        cache = KVCache.from_hf_output(past_key_values)
        assert cache.device == torch.device("cpu")

        # Move to CPU again (no-op)
        cache2 = cache.to(torch.device("cpu"))
        assert cache2.device == torch.device("cpu")


class TestKVCompatibility:
    """Test KV cache compatibility validation."""

    def test_compatible_caches(self):
        """Test that compatible caches don't raise errors."""
        num_layers = 2
        B, H, D = 1, 4, 16

        past_kv_base = tuple(
            (torch.randn(B, H, 10, D), torch.randn(B, H, 10, D))
            for _ in range(num_layers)
        )
        past_kv_new = tuple(
            (torch.randn(B, H, 5, D), torch.randn(B, H, 5, D))
            for _ in range(num_layers)
        )

        base_cache = KVCache.from_hf_output(past_kv_base)
        new_cache = KVCache.from_hf_output(past_kv_new)

        # Should not raise
        validate_kv_compatibility(base_cache, new_cache)

    def test_incompatible_num_layers(self):
        """Test that different layer counts raise error."""
        B, H, D = 1, 4, 16

        past_kv_base = tuple(
            (torch.randn(B, H, 10, D), torch.randn(B, H, 10, D)) for _ in range(3)
        )
        past_kv_new = tuple(
            (torch.randn(B, H, 5, D), torch.randn(B, H, 5, D)) for _ in range(2)
        )

        base_cache = KVCache.from_hf_output(past_kv_base)
        new_cache = KVCache.from_hf_output(past_kv_new)

        with pytest.raises(ValueError, match="Layer count mismatch"):
            validate_kv_compatibility(base_cache, new_cache)

    def test_incompatible_dtype(self):
        """Test that different dtypes raise error."""
        num_layers = 2
        B, H, D = 1, 4, 16

        past_kv_base = tuple(
            (torch.randn(B, H, 10, D), torch.randn(B, H, 10, D))
            for _ in range(num_layers)
        )
        past_kv_new = tuple(
            (
                torch.randn(B, H, 5, D, dtype=torch.float16),
                torch.randn(B, H, 5, D, dtype=torch.float16),
            )
            for _ in range(num_layers)
        )

        base_cache = KVCache.from_hf_output(past_kv_base)
        new_cache = KVCache.from_hf_output(past_kv_new)

        with pytest.raises(ValueError, match="Dtype mismatch"):
            validate_kv_compatibility(base_cache, new_cache)

    def test_incompatible_shapes(self):
        """Test that shape mismatches raise error."""
        num_layers = 2
        B, D = 1, 16

        # Different number of heads
        past_kv_base = tuple(
            (torch.randn(B, 4, 10, D), torch.randn(B, 4, 10, D))
            for _ in range(num_layers)
        )
        past_kv_new = tuple(
            (torch.randn(B, 8, 5, D), torch.randn(B, 8, 5, D))
            for _ in range(num_layers)
        )

        base_cache = KVCache.from_hf_output(past_kv_base)
        new_cache = KVCache.from_hf_output(past_kv_new)

        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_kv_compatibility(base_cache, new_cache)
