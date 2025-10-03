"""
Mock-based tests for Speculative Decoding Pipeline

These tests use mocks to avoid memory issues when loading large models.
Tests the logic and structure without actually running inference.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec.draft_model import DraftModel  # noqa: E402
from specdec.pipeline import SpeculativePipeline  # noqa: E402
from specdec.verifier import Verifier  # noqa: E402


class TestSpeculativePipelineMock:
    """Mock-based test cases for the speculative decoding pipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch("specdec.pipeline.DraftModel") as mock_draft, patch(
            "specdec.pipeline.Verifier"
        ) as mock_verifier:

            # Mock the models
            mock_draft_instance = Mock()
            mock_verifier_instance = Mock()

            # Mock tokenizer info
            mock_draft_instance.get_tokenizer_info.return_value = {
                "model_name": "distilgpt2",
                "vocab_size": 50257,
                "pad_token_id": 50256,
                "eos_token_id": 50256,
            }
            mock_verifier_instance.get_tokenizer_info.return_value = {
                "model_name": "facebook/opt-125m",
                "vocab_size": 50265,
                "pad_token_id": 1,
                "eos_token_id": 2,
            }

            # Mock compatibility check
            mock_draft_instance.check_tokenizer_compatibility.return_value = False

            # Mock generation results
            mock_draft_instance.generate_draft_tokens.return_value = (
                torch.tensor([[1, 2, 3, 4]]),  # draft tokens
                torch.randn(1, 4, 50257),  # draft logits
            )

            mock_verifier_instance.verify_tokens.return_value = {
                "accepted_len": 2,
                "accepted_tokens": torch.tensor([[1, 2]]),
                "verification_time_ms": 10.0,
                "base_tokens": torch.tensor([[1, 2, 5, 6]]),
                "proposed_tokens": torch.tensor([[1, 2, 3, 4]]),
            }

            mock_verifier_instance.generate_fallback_token.return_value = {
                "fallback_token": torch.tensor([[7]]),
                "generation_time_ms": 5.0,
            }

            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
            mock_tokenizer.decode.return_value = "Hello world"
            mock_tokenizer.eos_token_id = 2

            mock_verifier_instance.tokenizer = mock_tokenizer

            # Set up the pipeline
            pipeline = SpeculativePipeline(
                base_model="facebook/opt-125m",
                draft_model="distilgpt2",
                max_draft=4,
                device="cpu",
                seed=42,
            )

            # Replace with mocks
            pipeline.draft_model = mock_draft_instance
            pipeline.verifier = mock_verifier_instance

            return pipeline

    def test_pipeline_initialization_mock(self, mock_pipeline):
        """Test that pipeline initializes correctly with mocks."""
        assert mock_pipeline.max_draft == 4
        assert mock_pipeline.device == "cpu"
        assert mock_pipeline.config["seed"] == 42
        assert mock_pipeline.draft_model is not None
        assert mock_pipeline.verifier is not None

    def test_generate_basic_mock(self, mock_pipeline):
        """Test basic text generation with mocks."""
        prompt = "Hello world"
        result = mock_pipeline.generate(prompt, max_tokens=16)

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

    def test_config_loading_mock(self):
        """Test configuration loading from file."""
        config_path = PROJECT_ROOT / "configs" / "specdec.yaml"
        if config_path.exists():
            with patch("specdec.pipeline.DraftModel"), patch(
                "specdec.pipeline.Verifier"
            ):
                pipeline = SpeculativePipeline(config_path=str(config_path))
                assert pipeline.config["base_model"] == "facebook/opt-125m"
                assert pipeline.config["draft_model"] == "distilgpt2"
                assert pipeline.config["max_draft"] == 4

    def test_parameter_overrides_mock(self):
        """Test that parameters can be overridden."""
        with patch("specdec.pipeline.DraftModel"), patch("specdec.pipeline.Verifier"):
            pipeline = SpeculativePipeline(
                base_model="facebook/opt-125m",
                draft_model="distilgpt2",
                max_draft=2,  # Override default
                device="cpu",
                seed=123,
            )

            assert pipeline.max_draft == 2
            assert pipeline.config["seed"] == 123


class TestDraftModelMock:
    """Mock-based test cases for the draft model."""

    @pytest.fixture
    def mock_draft_model(self):
        """Create a mock draft model for testing."""
        with patch("specdec.draft_model.AutoModelForCausalLM") as mock_model, patch(
            "specdec.draft_model.AutoTokenizer"
        ) as mock_tokenizer:

            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.vocab_size = 50257
            mock_tokenizer_instance.pad_token_id = 50256
            mock_tokenizer_instance.eos_token_id = 50256
            mock_tokenizer_instance.bos_token_id = 50256
            mock_tokenizer_instance.unk_token_id = 50256
            mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3]])

            # Mock model
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance

            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            draft_model = DraftModel(model_name="distilgpt2", device="cpu")
            return draft_model

    def test_draft_model_initialization_mock(self, mock_draft_model):
        """Test draft model initialization with mocks."""
        assert mock_draft_model.model_name == "distilgpt2"
        assert mock_draft_model.device == "cpu"

    def test_tokenizer_info_mock(self, mock_draft_model):
        """Test tokenizer information retrieval with mocks."""
        info = mock_draft_model.get_tokenizer_info()

        assert "model_name" in info
        assert "vocab_size" in info
        assert "pad_token_id" in info
        assert "eos_token_id" in info
        assert info["model_name"] == "distilgpt2"


class TestVerifierMock:
    """Mock-based test cases for the verifier."""

    @pytest.fixture
    def mock_verifier(self):
        """Create a mock verifier for testing."""
        with patch("specdec.verifier.AutoModelForCausalLM") as mock_model, patch(
            "specdec.verifier.AutoTokenizer"
        ) as mock_tokenizer:

            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.vocab_size = 50265
            mock_tokenizer_instance.pad_token_id = 1
            mock_tokenizer_instance.eos_token_id = 2
            mock_tokenizer_instance.bos_token_id = 2
            mock_tokenizer_instance.unk_token_id = 2
            mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3]])

            # Mock model
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance

            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_model.from_pretrained.return_value = mock_model_instance

            verifier = Verifier(model_name="facebook/opt-125m", device="cpu")
            return verifier

    def test_verifier_initialization_mock(self, mock_verifier):
        """Test verifier initialization with mocks."""
        assert mock_verifier.model_name == "facebook/opt-125m"
        assert mock_verifier.device == "cpu"

    def test_find_accepted_length_mock(self, mock_verifier):
        """Test accepted length calculation with mocks."""
        # Test exact match
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[1, 2, 3, 5]])  # First 3 match
        accepted_len = mock_verifier._find_accepted_length(proposed, base)
        assert accepted_len == 3

        # Test no match
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[5, 6, 7, 8]])  # No matches
        accepted_len = mock_verifier._find_accepted_length(proposed, base)
        assert accepted_len == 0

        # Test partial match
        proposed = torch.tensor([[1, 2, 3, 4]])
        base = torch.tensor([[1, 2]])  # Only first 2 match
        accepted_len = mock_verifier._find_accepted_length(proposed, base)
        assert accepted_len == 2

    def test_tokenizer_info_mock(self, mock_verifier):
        """Test tokenizer information retrieval with mocks."""
        info = mock_verifier.get_tokenizer_info()

        assert "model_name" in info
        assert "vocab_size" in info
        assert "pad_token_id" in info
        assert "eos_token_id" in info
        assert info["model_name"] == "facebook/opt-125m"
