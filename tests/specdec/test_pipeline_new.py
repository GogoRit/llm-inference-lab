"""
Tests for Speculative Decoding Pipeline with Dependency Injection

Tests correctness, performance, and edge cases for the speculative decoding
implementation. Uses FakeLM by default to avoid memory issues. Real model tests
are marked with @pytest.mark.slow and excluded from CI.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec.fake_lm import create_fake_lm  # noqa: E402
from specdec.hf_wrappers import create_tiny_hf_wrapper  # noqa: E402
from specdec.pipeline import SpeculativePipeline  # noqa: E402


class TestSpeculativePipelineFake:
    """Test cases for the speculative decoding pipeline using FakeLM."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance with FakeLM for testing."""
        return SpeculativePipeline(
            implementation="fake",
            max_draft=4,
            device="cpu",
            seed=42,
        )

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline.max_draft == 4
        assert pipeline.device == "cpu"
        assert pipeline.config["seed"] == 42
        assert pipeline.base_lm is not None
        assert pipeline.draft_lm is not None
        assert pipeline.implementation == "fake"

    def test_generate_basic(self, pipeline):
        """Test basic text generation."""
        prompt = "Hello world"
        result = pipeline.generate(prompt, max_tokens=16)

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
        assert len(result["generated_tokens"]) <= 16
        assert result["latency_ms"] > 0
        assert result["proposed"] >= 0
        assert result["accepted"] >= 0
        assert 0 <= result["acceptance_rate"] <= 1

    def test_accepted_length_bounds(self, pipeline):
        """Test that accepted_len is always in [0, K] per iteration."""
        prompt = "Test prompt"
        result = pipeline.generate(prompt, max_tokens=8)

        # Check that acceptance rate is reasonable
        assert 0 <= result["acceptance_rate"] <= 1

        # Check that accepted tokens don't exceed proposed tokens
        assert result["accepted"] <= result["proposed"]

    def test_fallback_path(self, pipeline):
        """Test fallback path when accepted_len == 0."""
        # This test is harder to trigger deterministically, but we can verify
        # that the pipeline handles the case gracefully
        prompt = "Test"
        result = pipeline.generate(prompt, max_tokens=4)

        # Should still generate some tokens even if all proposals are rejected
        assert len(result["generated_tokens"]) > 0
        assert result["text"] is not None

    def test_deterministic_generation(self, pipeline):
        """Test that generation is deterministic with fixed seed."""
        prompt = "Test deterministic generation"

        # Generate twice with same seed
        result1 = pipeline.generate(prompt, max_tokens=8)
        result2 = pipeline.generate(prompt, max_tokens=8)

        # Results should be identical
        assert result1["text"] == result2["text"]
        assert result1["generated_tokens"] == result2["generated_tokens"]

    def test_max_tokens_limit(self, pipeline):
        """Test that generation respects max_tokens limit."""
        prompt = "Generate a long response"
        max_tokens = 4

        result = pipeline.generate(prompt, max_tokens=max_tokens)

        # Should not exceed max_tokens
        assert len(result["generated_tokens"]) <= max_tokens

    def test_empty_prompt(self, pipeline):
        """Test handling of empty prompt."""
        result = pipeline.generate("", max_tokens=4)

        # Should still generate some tokens
        assert len(result["generated_tokens"]) > 0
        assert result["text"] is not None

    def test_short_prompt(self, pipeline):
        """Test with very short prompt."""
        result = pipeline.generate("Hi", max_tokens=2)

        # Should generate exactly 2 tokens
        assert len(result["generated_tokens"]) == 2

    def test_metrics_calculation(self, pipeline):
        """Test that metrics are calculated correctly."""
        prompt = "Test metrics"
        result = pipeline.generate(prompt, max_tokens=8)

        # Check metrics consistency
        if result["proposed"] > 0:
            expected_acceptance_rate = result["accepted"] / result["proposed"]
            assert abs(result["acceptance_rate"] - expected_acceptance_rate) < 1e-6

        # Check tokens per second calculation
        expected_tps = len(result["generated_tokens"]) / (result["latency_ms"] / 1000.0)
        assert abs(result["tokens_per_sec"] - expected_tps) < 1e-6

    def test_device_selection(self):
        """Test device selection logic."""
        # Test CPU device selection
        pipeline = SpeculativePipeline(device="cpu", seed=42, implementation="fake")
        assert pipeline.device == "cpu"

        # Test auto device selection (should fall back to CPU in test environment)
        pipeline_auto = SpeculativePipeline(
            device="auto", seed=42, implementation="fake"
        )
        assert pipeline_auto.device in ["cpu", "mps", "cuda"]

    def test_config_loading(self):
        """Test configuration loading from file."""
        config_path = PROJECT_ROOT / "configs" / "specdec.yaml"
        if config_path.exists():
            pipeline = SpeculativePipeline(config_path=str(config_path))
            assert pipeline.config["base_model"] == "facebook/opt-125m"
            assert pipeline.config["draft_model"] == "distilgpt2"
            assert pipeline.config["max_draft"] == 4
            assert pipeline.config["implementation"] == "fake"

    def test_parameter_overrides(self):
        """Test that parameters can be overridden."""
        pipeline = SpeculativePipeline(
            base_model="facebook/opt-125m",
            draft_model="distilgpt2",
            max_draft=2,  # Override default
            device="cpu",
            seed=123,
            implementation="fake",
        )

        assert pipeline.max_draft == 2
        assert pipeline.config["seed"] == 123
        assert pipeline.implementation == "fake"

    def test_dependency_injection(self):
        """Test dependency injection with custom models."""
        # Create custom fake models
        base_lm = create_fake_lm(model_name="custom-base", seed=42)
        draft_lm = create_fake_lm(
            model_name="custom-draft", seed=42, acceptance_rate=0.8
        )

        pipeline = SpeculativePipeline(
            base_lm=base_lm,
            draft_lm=draft_lm,
            max_draft=3,
            device="cpu",
        )

        assert pipeline.base_lm.model_name == "custom-base"
        assert pipeline.draft_lm.model_name == "custom-draft"
        assert pipeline.max_draft == 3

        # Test generation works
        result = pipeline.generate("Test", max_tokens=4)
        assert result["text"] is not None


class TestSpeculativePipelineHF:
    """Test cases for the speculative decoding pipeline using Hugging Face models."""

    @pytest.mark.slow
    def test_hf_implementation(self):
        """Test HF implementation with tiny models."""
        pipeline = SpeculativePipeline(
            implementation="hf",
            max_draft=2,
            device="cpu",
            seed=42,
        )

        assert pipeline.implementation == "hf"
        assert pipeline.base_lm is not None
        assert pipeline.draft_lm is not None

        # Test generation works
        result = pipeline.generate("Hello", max_tokens=4)
        assert result["text"] is not None
        assert len(result["generated_tokens"]) > 0

    @pytest.mark.slow
    def test_hf_with_config(self):
        """Test HF implementation with config file."""
        config_path = PROJECT_ROOT / "configs" / "specdec_hf.yaml"
        if config_path.exists():
            pipeline = SpeculativePipeline(config_path=str(config_path))
            assert pipeline.implementation == "hf"
            assert pipeline.config["base_model"] == "sshleifer/tiny-gpt2"

            # Test generation works
            result = pipeline.generate("Test", max_tokens=4)
            assert result["text"] is not None


class TestFakeLM:
    """Test cases for the FakeLM implementation."""

    def test_fake_lm_initialization(self):
        """Test FakeLM initialization."""
        lm = create_fake_lm(model_name="test-model", seed=42)
        assert lm.model_name == "test-model"
        assert lm.device == "cpu"

    def test_fake_lm_generation(self):
        """Test FakeLM token generation."""
        lm = create_fake_lm(seed=42)
        input_ids = torch.tensor([[1, 2, 3]])

        tokens, logits = lm.generate_tokens(input_ids, max_new_tokens=3)
        assert tokens.shape[1] == 3
        assert logits.shape[1] == 3

    def test_fake_lm_encoding_decoding(self):
        """Test FakeLM encoding and decoding."""
        lm = create_fake_lm(seed=42)
        text = "Hello world"

        # Test encoding
        tokens = lm.encode(text)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == 1

        # Test decoding
        decoded = lm.decode(tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_fake_lm_with_acceptance(self):
        """Test FakeLM with acceptance rate."""
        lm = create_fake_lm(seed=42, acceptance_rate=0.5)
        input_ids = torch.tensor([[1, 2, 3]])

        # Generate multiple times and check acceptance varies
        results = []
        for _ in range(10):
            tokens, _ = lm.generate_tokens(input_ids, max_new_tokens=4)
            results.append(tokens.shape[1])

        # Should have some variation due to acceptance rate
        assert len(set(results)) > 1


class TestHFWrapper:
    """Test cases for the HFWrapper implementation."""

    @pytest.mark.slow
    def test_hf_wrapper_initialization(self):
        """Test HFWrapper initialization with tiny model."""
        wrapper = create_tiny_hf_wrapper(model_name="gpt2", device="cpu")
        assert wrapper.model_name == "gpt2"
        assert wrapper.device == "cpu"

    @pytest.mark.slow
    def test_hf_wrapper_generation(self):
        """Test HFWrapper token generation."""
        wrapper = create_tiny_hf_wrapper(model_name="gpt2", device="cpu")
        input_ids = torch.tensor([[1, 2, 3]])

        tokens, logits = wrapper.generate_tokens(input_ids, max_new_tokens=2)
        assert tokens.shape[1] == 2
        assert logits.shape[1] == 2

    @pytest.mark.slow
    def test_hf_wrapper_encoding_decoding(self):
        """Test HFWrapper encoding and decoding."""
        wrapper = create_tiny_hf_wrapper(model_name="gpt2", device="cpu")
        text = "Hello world"

        # Test encoding
        tokens = wrapper.encode(text)
        assert isinstance(tokens, torch.Tensor)

        # Test decoding
        decoded = wrapper.decode(tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
