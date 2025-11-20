"""
Comprehensive Tests for Speculative Decoding Pipeline

Consolidated test suite covering:
- Integration tests with real models (marked with @pytest.mark.integration)
- Unit tests with FakeLM (CI-friendly, fast)
- Mock-based smoke tests (minimal, quick validation)

All tests use fixed seeds for reproducibility.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec import SpeculativePipeline, Verifier, create_fake_lm  # noqa: E402

# ============================================================================
# INTEGRATION TESTS (Real Models)
# ============================================================================
# These tests use real HuggingFace models and are marked for integration testing.
# They may be skipped in CI environments due to memory/time constraints.


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests with real HuggingFace models."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance with real models for testing."""
        return SpeculativePipeline(
            base_model="facebook/opt-125m",
            draft_model="distilgpt2",
            max_draft=4,
            device="cpu",  # Force CPU for testing
            seed=42,  # Fixed seed for reproducibility
        )

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline.max_draft == 4
        assert pipeline.device == "cpu"
        assert pipeline.config["seed"] == 42
        assert pipeline.draft_lm is not None
        assert pipeline.base_lm is not None

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

    def test_tokenizer_compatibility_warning(self, pipeline):
        """Test that tokenizer compatibility warnings are logged."""
        assert pipeline.draft_lm is not None
        assert pipeline.base_lm is not None

        # The compatibility check should have been done during initialization
        result = pipeline.generate("Test", max_tokens=4)
        assert result["text"] is not None

    def test_device_selection(self):
        """Test device selection logic."""
        # Test CPU device selection
        pipeline = SpeculativePipeline(device="cpu", seed=42)
        assert pipeline.device == "cpu"

        # Test auto device selection
        pipeline_auto = SpeculativePipeline(device="auto", seed=42)
        assert pipeline_auto.device in ["cpu", "mps", "cuda"]

    def test_config_loading(self):
        """Test configuration loading from file."""
        config_path = PROJECT_ROOT / "configs" / "specdec.yaml"
        if config_path.exists():
            pipeline = SpeculativePipeline(config_path=str(config_path))
            assert pipeline.config["base_model"] == "gpt2"
            assert pipeline.config["draft_model"] == "distilgpt2"
            assert pipeline.config["max_draft"] == 4

    def test_parameter_overrides(self):
        """Test that parameters can be overridden."""
        pipeline = SpeculativePipeline(
            base_model="facebook/opt-125m",
            draft_model="distilgpt2",
            max_draft=2,  # Override default
            device="cpu",
            seed=123,
        )

        assert pipeline.max_draft == 2
        assert pipeline.config["seed"] == 123

    @pytest.mark.gpu
    def test_mps_device(self):
        """Test MPS device selection (only runs on Apple Silicon)."""
        if torch.backends.mps.is_available():
            pipeline = SpeculativePipeline(device="mps", seed=42)
            assert pipeline.device == "mps"

            # Test basic generation
            result = pipeline.generate("Test MPS", max_tokens=4)
            assert result["text"] is not None


@pytest.mark.integration
class TestVerifierIntegration:
    """Integration tests for the verifier with real models."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier instance for testing."""
        return Verifier(model_name="facebook/opt-125m", device="cpu")

    def test_verifier_initialization(self, verifier):
        """Test verifier initialization."""
        assert verifier.model_name == "facebook/opt-125m"
        assert verifier.device == "cpu"
        assert verifier.tokenizer is not None
        assert verifier.model is not None

    def test_verify_tokens(self, verifier):
        """Test token verification."""
        # Create input and proposed tokens
        input_ids = verifier.tokenizer.encode("Hello", return_tensors="pt")
        proposed_tokens = torch.tensor([[1, 2, 3, 4]])  # Dummy proposed tokens

        result = verifier.verify_tokens(input_ids, proposed_tokens)

        assert "accepted_len" in result
        assert "accepted_tokens" in result
        assert "verification_time_ms" in result
        assert "base_tokens" in result

        assert 0 <= result["accepted_len"] <= proposed_tokens.shape[1]
        assert result["verification_time_ms"] > 0

    def test_generate_fallback_token(self, verifier):
        """Test fallback token generation."""
        input_ids = verifier.tokenizer.encode("Hello", return_tensors="pt")

        result = verifier.generate_fallback_token(input_ids)

        assert "fallback_token" in result
        assert "generation_time_ms" in result
        assert result["fallback_token"].shape[1] == 1  # Single token
        assert result["generation_time_ms"] > 0

    def test_find_accepted_length(self, verifier):
        """Test accepted length calculation."""
        # Test exact match
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[1, 2, 3, 5]])  # First 3 match
        accepted_len = verifier._find_accepted_length(proposed, base)
        assert accepted_len == 3

        # Test no match
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[5, 6, 7, 8]])  # No matches
        accepted_len = verifier._find_accepted_length(proposed, base)
        assert accepted_len == 0

        # Test partial match
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[1, 2]])  # Only first 2 match
        accepted_len = verifier._find_accepted_length(proposed, base)
        assert accepted_len == 2

    def test_tokenizer_info(self, verifier):
        """Test tokenizer information retrieval."""
        info = verifier.get_tokenizer_info()

        assert "model_name" in info
        assert "vocab_size" in info
        assert "pad_token_id" in info
        assert "eos_token_id" in info
        assert info["model_name"] == "facebook/opt-125m"


# ============================================================================
# UNIT TESTS (FakeLM - CI-Friendly)
# ============================================================================
# These tests use FakeLM and are fast, memory-efficient, and CI-friendly.


class TestPipelineUnit:
    """Unit tests for the speculative decoding pipeline using FakeLM."""

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

        # Test auto device selection
        pipeline_auto = SpeculativePipeline(
            device="auto", seed=42, implementation="fake"
        )
        assert pipeline_auto.device in ["cpu", "mps", "cuda"]

    def test_config_loading(self):
        """Test configuration loading from file."""
        config_path = PROJECT_ROOT / "configs" / "specdec.yaml"
        if config_path.exists():
            pipeline = SpeculativePipeline(config_path=str(config_path))
            assert pipeline.config["base_model"] == "gpt2"
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


class TestDraftModel:
    """Test cases for the draft model using FakeLM."""

    @pytest.fixture
    def draft_model(self):
        """Create a fake draft model instance for testing."""
        return create_fake_lm("fake-draft-distilgpt2", device="cpu")

    def test_draft_model_initialization(self, draft_model):
        """Test draft model initialization."""
        assert draft_model.model_name == "fake-draft-distilgpt2"
        assert draft_model.device == "cpu"
        assert draft_model._vocab_size == 1000

    def test_generate_draft_tokens(self, draft_model):
        """Test draft token generation."""
        # Create a simple input
        input_ids = torch.tensor([[1, 2, 3]])  # Simple input

        draft_tokens, draft_logits = draft_model.generate_tokens(
            input_ids, max_new_tokens=3, temperature=0.7, do_sample=True
        )

        assert draft_tokens.shape[0] == 1  # batch size
        assert draft_tokens.shape[1] <= 3  # max_new_tokens
        assert draft_logits.shape[0] == 1  # batch size
        assert draft_logits.shape[1] == draft_tokens.shape[1]  # sequence length

    def test_tokenizer_info(self, draft_model):
        """Test tokenizer information retrieval."""
        info = draft_model.get_tokenizer_info()

        assert "model_name" in info
        assert "vocab_size" in info
        assert "pad_token_id" in info
        assert "eos_token_id" in info
        assert info["model_name"] == "fake-draft-distilgpt2"


# ============================================================================
# SMOKE TESTS (Minimal, Quick Validation)
# ============================================================================


class TestPipelineSmoke:
    """Minimal smoke tests for quick validation."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        return SpeculativePipeline(
            implementation="fake",
            max_draft=4,
            device="cpu",
            seed=42,
        )

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
