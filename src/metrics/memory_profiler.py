"""
Memory profiling utilities for CUDA and CPU/MPS.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiler for CUDA and CPU/MPS."""

    def __init__(self):
        self.peak_memory_mb = 0
        self.initial_memory_mb = 0
        self.memory_samples = []
        self.device = None

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def reset_peak_memory(self) -> None:
        """Reset peak memory stats."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            self.initial_memory_mb = 0

        self.peak_memory_mb = self.initial_memory_mb
        self.memory_samples = []

    def record_memory(self) -> float:
        """Record current memory usage."""
        if self.device == "cuda":
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        elif self.device == "mps":
            # MPS doesn't have memory tracking, estimate from tensor sizes
            current_memory = self._estimate_mps_memory()
            peak_memory = current_memory
        else:
            # CPU - no GPU memory
            current_memory = 0
            peak_memory = 0

        self.memory_samples.append(current_memory)
        self.peak_memory_mb = max(self.peak_memory_mb, peak_memory)

        return current_memory

    def _estimate_mps_memory(self) -> float:
        """Estimate MPS memory usage (rough approximation)."""
        # This is a rough estimate - MPS doesn't provide detailed memory stats
        # We'll use a simple heuristic based on tensor sizes
        total_elements = 0
        # MPS doesn't have _get_all_tensors, use gc as fallback
        import gc

        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_mps:
                if hasattr(obj, "numel"):
                    total_elements += obj.numel()

        # Assume float16 (2 bytes per element)
        return total_elements * 2 / 1024 / 1024

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self.device == "cuda":
            return {
                "device": "cuda",
                "current_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                "reserved_memory_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "samples": self.memory_samples,
            }
        elif self.device == "mps":
            current_memory = self._estimate_mps_memory()
            return {
                "device": "mps",
                "current_memory_mb": current_memory,
                "peak_memory_mb": self.peak_memory_mb,
                "reserved_memory_mb": current_memory,  # Same as current for MPS
                "samples": self.memory_samples,
            }
        else:
            return {
                "device": "cpu",
                "current_memory_mb": 0,
                "peak_memory_mb": 0,
                "reserved_memory_mb": 0,
                "samples": self.memory_samples,
            }

    def log_memory_summary(self) -> None:
        """Log memory usage summary."""
        stats = self.get_memory_stats()
        logger.info(
            f"Memory usage ({stats['device']}): "
            f"current={stats['current_memory_mb']:.1f}MB, "
            f"peak={stats['peak_memory_mb']:.1f}MB"
        )


# Global memory profiler instance
_memory_profiler = None


def get_memory_profiler() -> MemoryProfiler:
    """Get global memory profiler instance."""
    global _memory_profiler
    if _memory_profiler is None:
        _memory_profiler = MemoryProfiler()
    return _memory_profiler


def reset_peak_memory() -> None:
    """Reset peak memory stats."""
    get_memory_profiler().reset_peak_memory()


def record_memory() -> float:
    """Record current memory usage."""
    return get_memory_profiler().record_memory()


def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics."""
    return get_memory_profiler().get_memory_stats()


def log_memory_summary() -> None:
    """Log memory usage summary."""
    get_memory_profiler().log_memory_summary()
