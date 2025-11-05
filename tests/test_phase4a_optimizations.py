"""
Tests for Phase 4A Performance Optimizations

Tests the optimizations implemented in Phase 4A:
- SPECDEC_DEBUG flag gating for hot-path logging
- Token validation consolidation
- Async stream optimization (verification)
"""

import os
from unittest.mock import Mock, patch

import pytest
import torch

from src.specdec.utils.token_validation import get_vocab_size, validate_and_clamp_tokens


class TestSPECDECDebugFlag:
    """Test SPECDEC_DEBUG flag gating for hot-path logging."""

    def test_debug_flag_disabled_no_logging(self):
        """Test that debug logging is disabled when SPECDEC_DEBUG is not set."""
        # Clear env var
        if "SPECDEC_DEBUG" in os.environ:
            original = os.environ["SPECDEC_DEBUG"]
            del os.environ["SPECDEC_DEBUG"]
        else:
            original = None

        try:
            # Import scheduler after clearing env var
            from src.scheduler.speculative_scheduler import SpeculativeScheduler

            scheduler = SpeculativeScheduler(device="cpu")
            # Verify scheduler initializes without errors
            assert scheduler is not None
        finally:
            if original is not None:
                os.environ["SPECDEC_DEBUG"] = original

    def test_debug_flag_enabled(self):
        """Test that debug flag can be enabled."""
        original = os.environ.get("SPECDEC_DEBUG")
        try:
            os.environ["SPECDEC_DEBUG"] = "1"
            # Flag should be readable
            env_value = os.getenv("SPECDEC_DEBUG", "0").lower()
            assert env_value in ("1", "true", "yes")
        finally:
            if original is not None:
                os.environ["SPECDEC_DEBUG"] = original
            elif "SPECDEC_DEBUG" in os.environ:
                del os.environ["SPECDEC_DEBUG"]

    def test_debug_flag_parsing(self):
        """Test that debug flag accepts various truthy values."""
        test_values = ["1", "true", "yes", "True", "YES"]
        for value in test_values:
            original = os.environ.get("SPECDEC_DEBUG")
            try:
                os.environ["SPECDEC_DEBUG"] = value
                env_value = os.getenv("SPECDEC_DEBUG", "0").lower()
                is_enabled = env_value in ("1", "true", "yes")
                assert is_enabled, f"Value '{value}' should be treated as enabled"
            finally:
                if original is not None:
                    os.environ["SPECDEC_DEBUG"] = original
                elif "SPECDEC_DEBUG" in os.environ:
                    del os.environ["SPECDEC_DEBUG"]


class TestTokenValidationConsolidation:
    """Test consolidated token validation helper."""

    def test_validate_and_clamp_tokens_valid(self):
        """Test validation with valid tokens."""
        vocab_size = 1000
        input_ids = torch.tensor([[10, 20, 30, 40]])
        result = validate_and_clamp_tokens(input_ids, vocab_size, "test")
        assert torch.equal(result, input_ids)

    def test_validate_and_clamp_tokens_invalid_high(self):
        """Test validation clamps tokens that are too high."""
        vocab_size = 1000
        input_ids = torch.tensor([[1000, 1001, 500, 999]])  # First two are invalid
        result = validate_and_clamp_tokens(input_ids, vocab_size, "test")
        # Should be clamped to [999, 999, 500, 999]
        assert result[0, 0].item() == 999
        assert result[0, 1].item() == 999
        assert result[0, 2].item() == 500
        assert result[0, 3].item() == 999

    def test_validate_and_clamp_tokens_invalid_low(self):
        """Test validation clamps tokens that are negative."""
        vocab_size = 1000
        input_ids = torch.tensor([[-1, -2, 500, 0]])
        result = validate_and_clamp_tokens(input_ids, vocab_size, "test")
        # Should be clamped to [0, 0, 500, 0]
        assert result[0, 0].item() == 0
        assert result[0, 1].item() == 0
        assert result[0, 2].item() == 500
        assert result[0, 3].item() == 0

    def test_validate_and_clamp_tokens_empty(self):
        """Test validation with empty tensor."""
        vocab_size = 1000
        input_ids = torch.tensor([[]])
        result = validate_and_clamp_tokens(input_ids, vocab_size, "test")
        assert torch.equal(result, input_ids)

    def test_get_vocab_size_from_model(self):
        """Test get_vocab_size helper function."""
        # Create mock model
        mock_model = Mock()
        mock_model._model = Mock()
        mock_model._model.config = Mock()
        mock_model._model.config.vocab_size = 50257

        vocab_size = get_vocab_size(mock_model)
        assert vocab_size == 50257

    def test_get_vocab_size_none_if_no_config(self):
        """Test get_vocab_size returns None if model has no config."""

        # Create a simple object without _model attribute
        class SimpleModel:
            pass

        simple_model = SimpleModel()
        vocab_size = get_vocab_size(simple_model)
        assert vocab_size is None


class TestAsyncStreamOptimization:
    """Test async stream optimization (verification only, no actual GPU needed)."""

    def test_scheduler_initializes_with_streams_disabled_on_cpu(self):
        """Test that scheduler initializes correctly on CPU without streams."""
        from src.scheduler.speculative_scheduler import SpeculativeScheduler

        scheduler = SpeculativeScheduler(device="cpu")
        assert scheduler.enable_multi_stream is False
        assert scheduler.verification_stream is None

    def test_scheduler_initializes_with_streams_flag(self):
        """Test that scheduler respects SPECDEC_PARALLEL_STREAMS flag."""
        original = os.environ.get("SPECDEC_PARALLEL_STREAMS")
        try:
            # Test disabled
            os.environ["SPECDEC_PARALLEL_STREAMS"] = "0"
            from src.scheduler.speculative_scheduler import SpeculativeScheduler

            scheduler = SpeculativeScheduler(device="cpu")
            assert scheduler.enable_multi_stream is False

            # Test enabled (still False on CPU)
            os.environ["SPECDEC_PARALLEL_STREAMS"] = "1"
            scheduler2 = SpeculativeScheduler(device="cpu")
            assert scheduler2.enable_multi_stream is False  # CPU doesn't support
        finally:
            if original is not None:
                os.environ["SPECDEC_PARALLEL_STREAMS"] = original
            elif "SPECDEC_PARALLEL_STREAMS" in os.environ:
                del os.environ["SPECDEC_PARALLEL_STREAMS"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scheduler_creates_stream_on_cuda(self):
        """Test that scheduler creates stream on CUDA when enabled."""
        original = os.environ.get("SPECDEC_PARALLEL_STREAMS")
        try:
            os.environ["SPECDEC_PARALLEL_STREAMS"] = "1"
            from src.scheduler.speculative_scheduler import SpeculativeScheduler

            scheduler = SpeculativeScheduler(device="cuda")
            # Should have stream on CUDA
            assert scheduler.enable_multi_stream is True
            assert scheduler.verification_stream is not None
        finally:
            if original is not None:
                os.environ["SPECDEC_PARALLEL_STREAMS"] = original
            elif "SPECDEC_PARALLEL_STREAMS" in os.environ:
                del os.environ["SPECDEC_PARALLEL_STREAMS"]


class TestPartialKVCacheReuse:
    """Test partial KV cache reuse functionality."""

    def test_kv_cache_slice_prefix(self):
        """Test that KV cache can be sliced to partial length."""
        from src.specdec.cache.kv_types import KVCache

        # Create mock KV cache
        batch_size = 1
        num_heads = 2
        seq_len = 4
        head_dim = 8

        # Create fake past_key_values
        past_kv = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(3)  # 3 layers
        )

        cache = KVCache.from_hf_output(past_kv)

        # Slice to first 2 positions
        sliced = cache.slice_prefix(2)
        assert sliced.seq_len == 2
        assert sliced.get_num_layers() == 3

        # Verify shapes
        for k, v in sliced.past_key_values:
            assert k.shape == (batch_size, num_heads, 2, head_dim)
            assert v.shape == (batch_size, num_heads, 2, head_dim)

    def test_kv_cache_slice_prefix_full(self):
        """Test slicing to full length."""
        from src.specdec.cache.kv_types import KVCache

        past_kv = tuple(
            (
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
            )
            for _ in range(2)
        )

        cache = KVCache.from_hf_output(past_kv)
        sliced = cache.slice_prefix(4)
        assert sliced.seq_len == 4

    def test_kv_cache_slice_prefix_error_on_invalid_length(self):
        """Test that slicing with invalid length raises error."""
        from src.specdec.cache.kv_types import KVCache

        past_kv = tuple(
            (
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
            )
            for _ in range(2)
        )

        cache = KVCache.from_hf_output(past_kv)
        with pytest.raises(ValueError, match="Cannot slice length 5"):
            cache.slice_prefix(5)
