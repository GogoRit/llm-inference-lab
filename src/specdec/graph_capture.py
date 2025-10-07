"""
CUDA Graph capture utilities for speculative decoding.
"""

import logging
import os
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


class GraphCapture:
    """CUDA Graph capture for speculative decoding."""

    def __init__(self, enabled: bool = None):
        """
        Initialize graph capture.

        Args:
            enabled: Whether graph capture is enabled. If None, checks SPECDEC_CUDA_GRAPH env var.
        """
        if enabled is None:
            enabled = os.getenv("SPECDEC_CUDA_GRAPH", "0") == "1"

        self.enabled = enabled and self.can_capture()
        self.graph = None
        self.captured = False
        self.sample_inputs = None

        if self.enabled:
            logger.info("CUDA Graph capture enabled")
        else:
            logger.debug("CUDA Graph capture disabled")

    def can_capture(self) -> bool:
        """Check if graph capture is possible."""
        if not torch.cuda.is_available():
            return False

        # Check if we're in training mode (graph capture requires eval mode)
        if torch.is_grad_enabled():
            logger.warning("Graph capture requires eval mode (gradients disabled)")
            return False

        return True

    def capture_once(self, fn: Callable, sample_inputs: Any) -> bool:
        """
        Capture CUDA graph for function.

        Args:
            fn: Function to capture
            sample_inputs: Sample inputs for capture

        Returns:
            True if capture successful, False otherwise
        """
        if not self.enabled or self.captured:
            return False

        try:
            logger.info("Capturing CUDA graph...")

            # Create graph
            self.graph = torch.cuda.CUDAGraph()

            # Capture the function
            with torch.cuda.graph(self.graph):
                _ = fn(sample_inputs)

            self.captured = True
            self.sample_inputs = sample_inputs
            logger.info("CUDA graph captured successfully")
            return True

        except Exception as e:
            logger.warning(f"CUDA graph capture failed: {e}")
            self.graph = None
            self.captured = False
            return False

    def replay(self) -> Any:
        """
        Replay captured graph.

        Returns:
            Result of graph execution
        """
        if not self.enabled or not self.captured or self.graph is None:
            raise RuntimeError("No graph captured or graph capture disabled")

        try:
            self.graph.replay()
            return self.sample_inputs  # Return the inputs as placeholder
        except Exception as e:
            logger.warning(f"CUDA graph replay failed: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if graph is ready for replay."""
        return self.enabled and self.captured and self.graph is not None

    def reset(self) -> None:
        """Reset graph capture state."""
        self.graph = None
        self.captured = False
        self.sample_inputs = None


# Global graph capture instance
_graph_capture = None


def get_graph_capture() -> GraphCapture:
    """Get global graph capture instance."""
    global _graph_capture
    if _graph_capture is None:
        _graph_capture = GraphCapture()
    return _graph_capture


def can_capture() -> bool:
    """Check if graph capture is possible."""
    return get_graph_capture().can_capture()


def capture_once(fn: Callable, sample_inputs: Any) -> bool:
    """Capture CUDA graph for function."""
    return get_graph_capture().capture_once(fn, sample_inputs)


def replay() -> Any:
    """Replay captured graph."""
    return get_graph_capture().replay()


def is_ready() -> bool:
    """Check if graph is ready for replay."""
    return get_graph_capture().is_ready()


def reset() -> None:
    """Reset graph capture state."""
    get_graph_capture().reset()
