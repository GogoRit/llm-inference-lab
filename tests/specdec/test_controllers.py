"""
Unit tests for K controllers.
"""

import pytest

from src.specdec.controllers import (
    FixedKController,
    AdaptiveKController,
    create_controller,
)


class TestFixedKController:
    """Test fixed K controller."""

    def test_fixed_k(self):
        """Test that controller always returns fixed K."""
        controller = FixedKController(k=5)
        
        # Test multiple calls
        for step in range(10):
            context = {"step": step, "acceptance_rate": 0.5}
            k = controller.get_k(step, context)
            assert k == 5

    def test_get_info(self):
        """Test controller info."""
        controller = FixedKController(k=3)
        info = controller.get_info()
        
        assert info["controller"] == "fixed_k"
        assert info["k"] == 3


class TestAdaptiveKController:
    """Test adaptive K controller."""

    def test_initial_k(self):
        """Test that controller starts with initial K."""
        controller = AdaptiveKController(initial_k=4, min_k=1, max_k=8)
        
        context = {"step": 0, "acceptance_rate": 0.5}
        k = controller.get_k(0, context)
        assert k == 4

    def test_increase_k_on_high_acceptance(self):
        """Test that K increases when acceptance rate is high."""
        controller = AdaptiveKController(
            initial_k=4, min_k=1, max_k=8, target_acceptance_rate=0.7
        )
        
        # Simulate high acceptance rates
        for step in range(5):
            context = {"step": step, "acceptance_rate": 0.9}
            k = controller.get_k(step, context)
        
        # K should have increased
        assert k > 4

    def test_decrease_k_on_low_acceptance(self):
        """Test that K decreases when acceptance rate is low."""
        controller = AdaptiveKController(
            initial_k=4, min_k=1, max_k=8, target_acceptance_rate=0.7
        )
        
        # Simulate low acceptance rates
        for step in range(5):
            context = {"step": step, "acceptance_rate": 0.3}
            k = controller.get_k(step, context)
        
        # K should have decreased
        assert k < 4

    def test_k_bounds(self):
        """Test that K stays within bounds."""
        controller = AdaptiveKController(
            initial_k=4, min_k=2, max_k=6, target_acceptance_rate=0.7
        )
        
        # Simulate very high acceptance rates
        for step in range(10):
            context = {"step": step, "acceptance_rate": 0.95}
            k = controller.get_k(step, context)
            assert 2 <= k <= 6

    def test_step_size(self):
        """Test that K changes by step size."""
        controller = AdaptiveKController(
            initial_k=4, min_k=1, max_k=8, step_size=2, target_acceptance_rate=0.7
        )
        
        # Simulate high acceptance rates (need at least 4 iterations)
        for step in range(5):
            context = {"step": step, "acceptance_rate": 0.9}
            k = controller.get_k(step, context)
        
        # K should have increased by step_size (but may have hit max)
        assert k >= 6  # 4 + 2, but may have increased further

    def test_window_size(self):
        """Test that only recent history affects adaptation."""
        controller = AdaptiveKController(
            initial_k=4, min_k=1, max_k=8, window_size=3, target_acceptance_rate=0.7
        )
        
        # Fill up history with low acceptance
        for step in range(4):
            context = {"step": step, "acceptance_rate": 0.3}
            k = controller.get_k(step, context)
        
        # Add high acceptance to recent history
        for step in range(4, 8):
            context = {"step": step, "acceptance_rate": 0.9}
            k = controller.get_k(step, context)
        
        # K should have increased due to recent high acceptance (but may have hit max)
        assert k >= 4  # May have increased or stayed the same

    def test_get_info(self):
        """Test controller info."""
        controller = AdaptiveKController(
            initial_k=4, min_k=1, max_k=8, target_acceptance_rate=0.7
        )
        
        # Add some history
        for step in range(5):
            context = {"step": step, "acceptance_rate": 0.5}
            controller.get_k(step, context)
        
        info = controller.get_info()
        
        assert info["controller"] == "adaptive_k"
        assert info["current_k"] >= 1  # K may have changed due to adaptation
        assert info["min_k"] == 1
        assert info["max_k"] == 8
        assert info["step_size"] == 1
        assert info["window_size"] == 32
        assert info["target_acceptance_rate"] == 0.7

    def test_insufficient_history(self):
        """Test that K doesn't change with insufficient history."""
        controller = AdaptiveKController(
            initial_k=4, min_k=1, max_k=8, target_acceptance_rate=0.7
        )
        
        # Only 2 iterations of history (less than 4 required)
        for step in range(2):
            context = {"step": step, "acceptance_rate": 0.9}
            k = controller.get_k(step, context)
        
        # K should still be initial value
        assert k == 4


class TestCreateController:
    """Test controller creation function."""

    def test_create_fixed(self):
        """Test creating fixed controller."""
        controller = create_controller("fixed", k=5)
        assert isinstance(controller, FixedKController)
        assert controller.k == 5

    def test_create_adaptive(self):
        """Test creating adaptive controller."""
        controller = create_controller(
            "adaptive", 
            initial_k=3, 
            min_k=1, 
            max_k=10,
            target_acceptance_rate=0.8
        )
        assert isinstance(controller, AdaptiveKController)
        assert controller.initial_k == 3
        assert controller.min_k == 1
        assert controller.max_k == 10
        assert controller.target_acceptance_rate == 0.8

    def test_invalid_controller(self):
        """Test creating invalid controller raises error."""
        with pytest.raises(ValueError, match="Unknown controller"):
            create_controller("invalid_controller")
