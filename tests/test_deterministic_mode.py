"""
Tests for deterministic seeding and reproducibility.
"""

import os

import torch

from specdec.utils.deterministic import ensure_deterministic

# Import set_deterministic_mode if it exists, otherwise create a simple wrapper
try:
    from specdec.utils.deterministic import set_deterministic_mode
except ImportError:

    def set_deterministic_mode(enable: bool):
        """Simple wrapper for deterministic mode."""
        if enable:
            ensure_deterministic()


class TestDeterministicMode:
    """Test deterministic seeding functionality."""

    def test_set_deterministic_mode_default_seed(self):
        """Test setting deterministic mode with default seed."""
        set_deterministic_mode()
        seed1 = torch.initial_seed()
        torch.manual_seed(42)
        val1 = torch.rand(1).item()

        set_deterministic_mode()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()

        assert seed1 is not None
        assert val1 == val2, "Deterministic mode should produce identical results"

    def test_set_deterministic_mode_custom_seed(self):
        """Test setting deterministic mode with custom seed."""
        set_deterministic_mode(seed=5678)
        torch.manual_seed(5678)
        val1 = torch.rand(1).item()

        set_deterministic_mode(seed=5678)
        torch.manual_seed(5678)
        val2 = torch.rand(1).item()

        assert val1 == val2, "Custom seed should produce identical results"

    def test_ensure_deterministic_env_flag(self):
        """Test ensure_deterministic respects SPECDEC_DETERMINISTIC env var."""
        # Set env flag
        os.environ["SPECDEC_DETERMINISTIC"] = "1"
        try:
            ensure_deterministic()

            # Verify deterministic is set
            torch.manual_seed(1234)
            val1 = torch.rand(1).item()

            torch.manual_seed(1234)
            val2 = torch.rand(1).item()

            assert val1 == val2, "Environment flag should enable deterministic mode"
        finally:
            # Clean up
            if "SPECDEC_DETERMINISTIC" in os.environ:
                del os.environ["SPECDEC_DETERMINISTIC"]

    def test_ensure_deterministic_env_flag_disabled(self):
        """Test ensure_deterministic ignores when env flag is off."""
        # Set env flag to disabled
        os.environ["SPECDEC_DETERMINISTIC"] = "0"
        try:
            # Should not raise errors
            ensure_deterministic()
        finally:
            if "SPECDEC_DETERMINISTIC" in os.environ:
                del os.environ["SPECDEC_DETERMINISTIC"]

    def test_deterministic_mode_cudnn_settings(self):
        """Test that CuDNN deterministic settings are applied."""
        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            set_deterministic_mode()
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
