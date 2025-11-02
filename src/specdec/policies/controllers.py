"""
K Controllers for Speculative Decoding

This module implements various strategies for controlling the number of draft tokens
(K) generated per step in speculative decoding.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class KController(ABC):
    """Abstract base class for K controllers."""

    @abstractmethod
    def get_k(self, step: int, context: Dict[str, Any]) -> int:
        """
        Determine the number of draft tokens to generate for this step.

        Args:
            step: Current step number (0-indexed)
            context: Context information (acceptance rates, performance, etc.)

        Returns:
            Number of draft tokens to generate (K)
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get controller information for logging."""
        pass


class FixedKController(KController):
    """Fixed K controller that always generates the same number of draft tokens."""

    def __init__(self, k: int = 4):
        """
        Initialize fixed K controller.

        Args:
            k: Fixed number of draft tokens to generate
        """
        self.k = k
        self.name = "fixed_k"

    def get_k(self, step: int, context: Dict[str, Any]) -> int:
        """Return fixed K value."""
        return self.k

    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        return {
            "controller": self.name,
            "k": self.k,
        }


class AdaptiveKController(KController):
    """Adaptive K controller that adjusts K based on acceptance rates."""

    def __init__(
        self,
        initial_k: int = 4,
        min_k: int = 1,
        max_k: int = 8,
        step_size: int = 1,
        window_size: int = 32,
        target_acceptance_rate: float = 0.7,
    ):
        """
        Initialize adaptive K controller.

        Args:
            initial_k: Initial K value
            min_k: Minimum K value
            max_k: Maximum K value
            step_size: Step size for K adjustments
            window_size: Window size for acceptance rate calculation
            target_acceptance_rate: Target acceptance rate for adaptation
        """
        self.initial_k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.step_size = step_size
        self.window_size = window_size
        self.target_acceptance_rate = target_acceptance_rate
        self.name = "adaptive_k"

        # State tracking
        self.current_k = initial_k
        self.acceptance_history: List[float] = []
        self.k_history: List[int] = []

    def get_k(self, step: int, context: Dict[str, Any]) -> int:
        """Return adaptive K value based on acceptance history."""
        # Update acceptance history if available
        if "acceptance_rate" in context:
            self.acceptance_history.append(context["acceptance_rate"])
            # Keep only recent history
            if len(self.acceptance_history) > self.window_size:
                self.acceptance_history = self.acceptance_history[-self.window_size :]

        # Adapt K based on recent acceptance rates
        if len(self.acceptance_history) >= 4:  # Need some history to adapt
            recent_acceptance = sum(self.acceptance_history[-4:]) / 4

            if recent_acceptance > self.target_acceptance_rate + 0.1:
                # High acceptance rate, increase K
                self.current_k = min(self.current_k + self.step_size, self.max_k)
            elif recent_acceptance < self.target_acceptance_rate - 0.1:
                # Low acceptance rate, decrease K
                self.current_k = max(self.current_k - self.step_size, self.min_k)

        # Record K history
        self.k_history.append(self.current_k)
        if len(self.k_history) > self.window_size:
            self.k_history = self.k_history[-self.window_size :]

        return self.current_k

    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        return {
            "controller": self.name,
            "current_k": self.current_k,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "step_size": self.step_size,
            "window_size": self.window_size,
            "target_acceptance_rate": self.target_acceptance_rate,
            "recent_acceptance_rate": (
                sum(self.acceptance_history[-4:]) / 4
                if len(self.acceptance_history) >= 4
                else None
            ),
        }


def create_controller(controller_type: str, **kwargs: Any) -> KController:
    """
    Create a K controller by type.

    Args:
        controller_type: Type of controller to create
        **kwargs: Controller-specific parameters

    Returns:
        KController instance

    Raises:
        ValueError: If controller_type is not recognized
    """
    if controller_type == "fixed":
        return FixedKController(kwargs.get("k", 4))
    elif controller_type == "adaptive":
        return AdaptiveKController(
            initial_k=kwargs.get("initial_k", 4),
            min_k=kwargs.get("min_k", 1),
            max_k=kwargs.get("max_k", 8),
            step_size=kwargs.get("step_size", 1),
            window_size=kwargs.get("window_size", 32),
            target_acceptance_rate=kwargs.get("target_acceptance_rate", 0.7),
        )
    else:
        raise ValueError(
            f"Unknown controller: {controller_type}. "
            f"Available: ['fixed', 'adaptive']"
        )
