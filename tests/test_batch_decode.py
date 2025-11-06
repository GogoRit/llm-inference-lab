"""
Tests for batch processing decode functionality and EOS token handling.

Verifies that:
1. Decode works correctly with token lists
2. EOS tokens are properly filtered out
3. Batch generation produces clean text output
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec.core.pipeline import SpeculativePipeline  # noqa: E402
from specdec.models.hf_wrappers import HFWrapper  # noqa: E402


class TestBatchDecode:
    """Test batch decode functionality and EOS handling."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        return SpeculativePipeline(
            base_model="gpt2",
            draft_model="distilgpt2",
            max_draft=4,
            device="cpu",  # Force CPU for testing
            seed=42,
        )

    def test_decode_with_token_list(self, pipeline):
        """Test that decode works correctly with token lists."""
        # Use HFWrapper for decode tests (FakeLM decode doesn't handle lists)
        if isinstance(pipeline.base_lm, HFWrapper):
            # Get a simple token list
            tokens = [464, 464, 56, 464]  # "The", "The", "U", "The"

            # Decode should work with list directly
            decoded = pipeline.base_lm.decode(tokens)

            # Should produce valid text
            assert isinstance(decoded, str)
            assert len(decoded) > 0
            # Should not contain special token markers
            assert "<|endoftext|>" not in decoded
        else:
            pytest.skip("HFWrapper needed for decode list tests")

    def test_decode_filters_eos_tokens(self, pipeline):
        """Test that decode filters out EOS tokens."""
        # Use HFWrapper for decode tests
        if isinstance(pipeline.base_lm, HFWrapper):
            # Get EOS token ID
            eos_token_id = 50256  # GPT-2 EOS token

            # Create token list with EOS in the middle
            tokens = [464, 464, eos_token_id, 56, 464]

            # Decode should skip EOS token
            decoded = pipeline.base_lm.decode(tokens)

            # Should not contain EOS token representation
            assert "<|endoftext|>" not in decoded
            # Decode with skip_special_tokens=True should filter it
        else:
            pytest.skip("HFWrapper needed for decode list tests")

    def test_batch_generate_produces_text(self, pipeline):
        """Test that batch generation produces valid text output."""
        prompts = ["Hello world", "Test prompt"]

        # Run batch generation
        results = pipeline.generate_batch(prompts, max_tokens=8)

        # Should return results for all prompts
        assert len(results) == len(prompts)

        # Each result should have text field
        for result in results:
            assert "text" in result
            assert isinstance(result["text"], str)
            # Text should not be empty (unless max_tokens was 0)
            if result.get("success", True):
                assert len(result["text"]) > 0 or result.get("generated_tokens", 0) == 0

    def test_batch_generate_no_eos_in_text(self, pipeline):
        """Test that batch generation doesn't include EOS tokens in text."""
        prompts = ["Generate a short response"]

        # Run batch generation
        results = pipeline.generate_batch(prompts, max_tokens=16)

        # Text should not contain EOS token representation
        for result in results:
            if result.get("text"):
                assert "<|endoftext|>" not in result["text"]
                # Text should be clean (no special token markers)
                assert result["text"].strip() == result["text"] or result[
                    "text"
                ].startswith("\n")

    def test_decode_with_empty_list(self, pipeline):
        """Test decode with empty token list."""
        if isinstance(pipeline.base_lm, HFWrapper):
            decoded = pipeline.base_lm.decode([])
            assert decoded == ""
        else:
            pytest.skip("HFWrapper needed for decode list tests")

    def test_decode_with_special_tokens_only(self, pipeline):
        """Test decode when only special tokens are present."""
        if isinstance(pipeline.base_lm, HFWrapper):
            eos_token_id = 50256
            tokens = [eos_token_id, eos_token_id]

            # Decode should return empty string (special tokens skipped)
            decoded = pipeline.base_lm.decode(tokens)
            # Should be empty or contain only whitespace
            assert len(decoded.strip()) == 0
        else:
            pytest.skip("HFWrapper needed for decode list tests")

    def test_hf_wrapper_decode_handles_lists(self):
        """Test that HFWrapper.decode correctly handles list input."""
        # Create a wrapper with a simple model
        wrapper = HFWrapper("gpt2", device="cpu")

        # Test with list
        tokens_list = [464, 464, 56]
        decoded_list = wrapper.decode(tokens_list)

        # Test with tensor
        tokens_tensor = torch.tensor([464, 464, 56])
        decoded_tensor = wrapper.decode(tokens_tensor)

        # Both should produce same result
        assert decoded_list == decoded_tensor
        assert isinstance(decoded_list, str)
        assert len(decoded_list) > 0
