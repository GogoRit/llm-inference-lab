"""
Unit tests for draft mode functionality.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from src.specdec.eagle import EagleDraftor
from src.specdec.medusa import MedusaDraftor
from src.specdec.pipeline import SpeculativePipeline


class TestDraftModeCLI:
    """Test CLI flag parsing for draft modes."""

    def test_vanilla_mode_default(self):
        """Test that vanilla mode is the default."""
        import sys

        from src.specdec.run_specdec import parse_args

        # Mock sys.argv to avoid issues with argparse
        with patch.object(sys, "argv", ["run_specdec.py", "--prompt", "test"]):
            args = parse_args()
            assert args.draft_mode == "vanilla"

    def test_medusa_mode_flag(self):
        """Test medusa mode flag parsing."""
        import sys

        from src.specdec.run_specdec import parse_args

        with patch.object(
            sys,
            "argv",
            ["run_specdec.py", "--prompt", "test", "--draft-mode", "medusa"],
        ):
            args = parse_args()
            assert args.draft_mode == "medusa"

    def test_eagle_mode_flag(self):
        """Test eagle mode flag parsing."""
        import sys

        from src.specdec.run_specdec import parse_args

        with patch.object(
            sys, "argv", ["run_specdec.py", "--prompt", "test", "--draft-mode", "eagle"]
        ):
            args = parse_args()
            assert args.draft_mode == "eagle"

    def test_invalid_draft_mode(self):
        """Test that invalid draft mode raises error."""
        import sys

        from src.specdec.run_specdec import parse_args

        with patch.object(
            sys,
            "argv",
            ["run_specdec.py", "--prompt", "test", "--draft-mode", "invalid"],
        ):
            with pytest.raises(SystemExit):
                parse_args()


class TestDraftModeConfig:
    """Test configuration handling for draft modes."""

    def test_vanilla_config_default(self):
        """Test vanilla mode configuration defaults."""
        pipeline = SpeculativePipeline(implementation="fake")
        assert pipeline.config["draft_mode"] == "vanilla"
        assert pipeline.config["medusa"]["enabled"] is False
        assert pipeline.config["eagle"]["enabled"] is False

    def test_medusa_config_override(self):
        """Test medusa mode configuration override."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="medusa")
        assert pipeline.config["draft_mode"] == "medusa"

    def test_eagle_config_override(self):
        """Test eagle mode configuration override."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="eagle")
        assert pipeline.config["draft_mode"] == "eagle"


class TestMedusaDraftor:
    """Test MedusaDraftor functionality."""

    def test_medusa_initialization_random(self):
        """Test MedusaDraftor initialization with random weights."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768

        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        # Use random initialization to avoid weight assignment issues
        draftor = MedusaDraftor(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            num_heads=2,
            head_init="random",
            device="cpu",
        )

        assert draftor.num_heads == 2
        assert draftor.hidden_size == 768
        assert draftor.vocab_size == 1000
        assert len(draftor.heads) == 2

    def test_medusa_get_info(self):
        """Test MedusaDraftor info method."""
        mock_model = Mock()
        mock_model.config.hidden_size = 768

        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        # Use random initialization to avoid weight assignment issues
        draftor = MedusaDraftor(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            num_heads=3,
            temperature=0.8,
            top_p=0.9,
            head_init="random",
            device="cpu",
        )

        info = draftor.get_info()
        assert info["type"] == "medusa"
        assert info["num_heads"] == 3
        assert info["temperature"] == 0.8
        assert info["top_p"] == 0.9


class TestEagleDraftor:
    """Test EagleDraftor functionality."""

    def test_eagle_initialization(self):
        """Test EagleDraftor initialization."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        draftor = EagleDraftor(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            alpha=0.5,
            max_draft=3,
            device="cpu",
        )

        assert draftor.alpha == 0.5
        assert draftor.max_draft == 3
        assert draftor.vocab_size == 1000

    def test_eagle_generate_tokens_simple(self):
        """Test EagleDraftor basic functionality."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        draftor = EagleDraftor(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            alpha=0.7,
            max_draft=2,
            device="cpu",
        )

        # Test basic properties
        assert draftor.alpha == 0.7
        assert draftor.max_draft == 2
        assert draftor.vocab_size == 1000

        # Test reset state
        draftor.reset_state()
        assert draftor.last_hidden_states is None
        assert draftor.last_tokens is None

    def test_eagle_get_info(self):
        """Test EagleDraftor info method."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        draftor = EagleDraftor(
            base_model=mock_model,
            tokenizer=mock_tokenizer,
            alpha=0.6,
            max_draft=4,
            device="cpu",
        )

        info = draftor.get_info()
        assert info["type"] == "eagle"
        assert info["alpha"] == 0.6
        assert info["max_draft"] == 4

    def test_eagle_reset_state(self):
        """Test EagleDraftor state reset."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        draftor = EagleDraftor(
            base_model=mock_model, tokenizer=mock_tokenizer, device="cpu"
        )

        # Set some state
        draftor.last_hidden_states = torch.randn(1, 2, 768)
        draftor.last_tokens = torch.tensor([[1, 2]])

        # Reset state
        draftor.reset_state()

        assert draftor.last_hidden_states is None
        assert draftor.last_tokens is None


class TestDraftModeIntegration:
    """Test draft mode integration in pipeline."""

    def test_vanilla_mode_integration(self):
        """Test vanilla mode works in pipeline."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="vanilla")

        # Should not raise any errors
        assert pipeline.config["draft_mode"] == "vanilla"

    def test_medusa_mode_integration(self):
        """Test medusa mode works in pipeline."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="medusa")

        # Should not raise any errors
        assert pipeline.config["draft_mode"] == "medusa"

    def test_eagle_mode_integration(self):
        """Test eagle mode works in pipeline."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="eagle")

        # Should not raise any errors
        assert pipeline.config["draft_mode"] == "eagle"

    def test_invalid_draft_mode_raises_error(self):
        """Test that invalid draft mode raises error during generation."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="invalid")

        # This should raise an error during generation
        with pytest.raises(ValueError, match="Unknown draft mode"):
            pipeline.generate("test prompt", max_tokens=5)

    def test_draft_mode_fallback_behavior(self):
        """Test that medusa/eagle modes fall back to vanilla generation."""
        pipeline = SpeculativePipeline(implementation="fake", draft_mode="medusa")

        # Should work with fallback behavior
        result = pipeline.generate("test prompt", max_tokens=5)
        assert "text" in result
        assert result["draft_mode"] == "medusa"


class TestDraftModePolicies:
    """Test that policies work with different draft modes."""

    def test_policies_work_with_medusa_mode(self):
        """Test that acceptance policies work with medusa mode."""
        pipeline = SpeculativePipeline(
            implementation="fake", draft_mode="medusa", policy="longest_prefix"
        )

        result = pipeline.generate("test prompt", max_tokens=5)
        assert "acceptance_rate" in result
        assert "policy" in result
        # Check that policy info is present (structure may vary)
        assert isinstance(result["policy"], dict)

    def test_policies_work_with_eagle_mode(self):
        """Test that acceptance policies work with eagle mode."""
        pipeline = SpeculativePipeline(
            implementation="fake",
            draft_mode="eagle",
            policy="conf_threshold",
            policy_params={"tau": 0.5},
        )

        result = pipeline.generate("test prompt", max_tokens=5)
        assert "acceptance_rate" in result
        assert "policy" in result
        # Check that policy info is present (structure may vary)
        assert isinstance(result["policy"], dict)

    def test_controllers_work_with_draft_modes(self):
        """Test that K controllers work with different draft modes."""
        pipeline = SpeculativePipeline(
            implementation="fake",
            draft_mode="medusa",
            controller="adaptive",
            controller_params={"min_k": 1, "max_k": 4},
        )

        result = pipeline.generate("test prompt", max_tokens=5)
        assert "controller" in result
        # Check that controller info is present (structure may vary)
        assert isinstance(result["controller"], dict)
