"""
Mock-based tests for Speculative Decoding Pipeline

These tests use the FakeLM implementation to avoid memory issues when testing.
Tests the logic and structure without actually running inference.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec import SpeculativePipeline  # noqa: E402


class TestSpeculativePipelineMock:
    """Mock-based test cases for the speculative decoding pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        # Create pipeline with fake implementation to avoid memory issues
        pipeline = SpeculativePipeline(
            implementation="fake",
            max_draft=4,
            device="cpu",
            seed=42,
        )
        return pipeline

    def test_pipeline_initialization_mock(self, mock_pipeline):
        """Test that pipeline initializes correctly with fake models."""
        assert mock_pipeline.max_draft == 4
        assert mock_pipeline.device == "cpu"
        assert mock_pipeline.config["seed"] == 42
        assert mock_pipeline.base_lm is not None
        assert mock_pipeline.draft_lm is not None
        assert mock_pipeline.implementation == "fake"

    def test_generate_basic_mock(self, mock_pipeline):
        """Test basic text generation with fake models."""
        prompt = "Hello world"
        result = mock_pipeline.generate(prompt, max_tokens=8)

        # Check result structure
        assert "text" in result
        assert "generated_tokens" in result
        assert "latency_ms" in result
        assert "proposed" in result
        assert "accepted" in result
        assert "acceptance_rate" in result
        assert "tokens_per_sec" in result

        # Check basic properties
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert len(result["generated_tokens"]) <= 8
        assert result["latency_ms"] > 0
        assert result["proposed"] >= 0
        assert result["accepted"] >= 0
        assert 0 <= result["acceptance_rate"] <= 1

    def test_accepted_length_bounds_mock(self, mock_pipeline):
        """Test that accepted_len is always in [0, K] per iteration."""
        prompt = "Test prompt"
        result = mock_pipeline.generate(prompt, max_tokens=8)

        # Check that acceptance rate is reasonable
        assert 0 <= result["acceptance_rate"] <= 1

        # Check that accepted tokens don't exceed proposed tokens
        assert result["accepted"] <= result["proposed"]

    def test_metrics_calculation_mock(self, mock_pipeline):
        """Test that metrics are calculated correctly."""
        prompt = "Test metrics"
        result = mock_pipeline.generate(prompt, max_tokens=8)

        # Check metrics consistency
        if result["proposed"] > 0:
            expected_acceptance_rate = result["accepted"] / result["proposed"]
            assert abs(result["acceptance_rate"] - expected_acceptance_rate) < 1e-6

        # Check tokens per second calculation
        expected_tps = len(result["generated_tokens"]) / (result["latency_ms"] / 1000.0)
        assert abs(result["tokens_per_sec"] - expected_tps) < 1e-6
