"""
Unit tests for verify_prefix kernel.
"""

# import numpy as np  # Unused import
import pytest
import torch

from src.kernels import get_kernel_info, verify_prefix
from src.kernels.reference import verify_prefix_ref


class TestVerifyPrefix:
    """Test verify_prefix kernel functionality."""

    def test_verify_prefix_basic(self):
        """Test basic verify_prefix functionality."""
        # Create test data
        B, K, V = 2, 3, 1000
        logits = torch.randn(B, K, V)
        draft_ids = torch.randint(0, V, (B, K))

        # Set up known matches
        logits[0, 0, draft_ids[0, 0]] = 10.0  # Match
        logits[0, 1, draft_ids[0, 1]] = 10.0  # Match
        logits[0, 2, draft_ids[0, 2] + 1] = 10.0  # No match

        logits[1, 0, draft_ids[1, 0] + 1] = 10.0  # No match

        # Test kernel
        accept_len, accepted_mask = verify_prefix(logits, draft_ids)

        # Verify results
        assert accept_len.shape == (B,)
        assert accepted_mask.shape == (B, K)
        assert accept_len[0] == 2  # First 2 positions match
        assert accept_len[1] == 0  # No matches
        assert accepted_mask[0, 0] == 1
        assert accepted_mask[0, 1] == 1
        assert accepted_mask[0, 2] == 0
        assert accepted_mask[1, 0] == 0

    def test_verify_prefix_edge_cases(self):
        """Test edge cases for verify_prefix."""
        # Test with K=1 and match
        B, K, V = 1, 1, 100
        logits = torch.randn(B, K, V)
        draft_ids = torch.randint(0, V, (B, K))

        # Set up match
        logits[0, 0, draft_ids[0, 0]] = 10.0

        accept_len, accepted_mask = verify_prefix(logits, draft_ids)

        assert accept_len[0] == 1
        assert accepted_mask[0, 0] == 1

        # Test with no matches (create new logits)
        logits_no_match = torch.randn(B, K, V)
        logits_no_match[0, 0, draft_ids[0, 0] + 1] = 10.0  # Different position

        accept_len, accepted_mask = verify_prefix(logits_no_match, draft_ids)

        assert accept_len[0] == 0
        assert accepted_mask[0, 0] == 0

    def test_verify_prefix_vs_reference(self):
        """Compare kernel output with reference implementation."""
        # Test multiple shapes
        test_cases = [
            (1, 1, 100),
            (1, 2, 1000),
            (2, 3, 5000),
            (4, 4, 10000),
        ]

        for B, K, V in test_cases:
            # Generate random test data
            logits = torch.randn(B, K, V)
            draft_ids = torch.randint(0, V, (B, K))

            # Test kernel
            accept_len_kernel, accepted_mask_kernel = verify_prefix(logits, draft_ids)

            # Test reference
            accept_len_ref, accepted_mask_ref = verify_prefix_ref(logits, draft_ids)

            # Compare results
            assert torch.equal(
                accept_len_kernel, accept_len_ref
            ), f"Accept length mismatch for shape {B}x{K}x{V}"
            assert torch.equal(
                accepted_mask_kernel, accepted_mask_ref
            ), f"Accepted mask mismatch for shape {B}x{K}x{V}"

    def test_verify_prefix_different_devices(self):
        """Test verify_prefix on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        B, K, V = 2, 3, 1000
        logits = torch.randn(B, K, V, device="cuda")
        draft_ids = torch.randint(0, V, (B, K), device="cuda")

        # Set up matches
        logits[0, 0, draft_ids[0, 0]] = 10.0
        logits[0, 1, draft_ids[0, 1]] = 10.0

        accept_len, accepted_mask = verify_prefix(logits, draft_ids)

        assert accept_len.device == logits.device
        assert accepted_mask.device == logits.device
        assert accept_len[0] == 2

    def test_verify_prefix_large_vocab(self):
        """Test verify_prefix with large vocabulary."""
        B, K, V = 1, 2, 50257  # GPT-2 vocab size
        logits = torch.randn(B, K, V)
        draft_ids = torch.randint(0, V, (B, K))

        # Set up matches
        logits[0, 0, draft_ids[0, 0]] = 10.0
        logits[0, 1, draft_ids[0, 1]] = 10.0

        accept_len, accepted_mask = verify_prefix(logits, draft_ids)

        assert accept_len[0] == 2
        assert accepted_mask[0, 0] == 1
        assert accepted_mask[0, 1] == 1

    def test_kernel_info(self):
        """Test kernel information retrieval."""
        info = get_kernel_info()

        assert "verify_backend" in info
        assert "kv_append_backend" in info
        assert "verify_available" in info
        assert "kv_append_available" in info

        # Verify backend should be one of the expected values
        assert info["verify_backend"] in [
            "cuda",
            "triton",
            "torch",
            "fallback",
            "unknown",
        ]
        assert info["kv_append_backend"] in [
            "cuda",
            "torch",
            "noop",
            "fallback",
            "unknown",
        ]


if __name__ == "__main__":
    pytest.main([__file__])
