"""
Memory profiling utilities for CUDA and CPU/MPS.
"""

import logging
from typing import Any, Dict

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
        """Estimate MPS memory usage using driver_allocated_memory if available."""
        # Try to use torch.mps.driver_allocated_memory() if available (PyTorch 2.0+)
        if hasattr(torch.mps, "driver_allocated_memory"):
            return torch.mps.driver_allocated_memory() / (1024**2)

        # Fallback: rough estimate based on tensor sizes
        total_elements = 0
        import gc

        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_mps:
                if hasattr(obj, "numel"):
                    total_elements += obj.numel()

        # Assume float16 (2 bytes per element)
        return total_elements * 2 / (1024**2)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics with enhanced CUDA/MPS tracking."""
        if self.device == "cuda":
            base_stats = {
                "device": "cuda",
                "current_memory_mb": torch.cuda.memory_allocated() / (1024**2),
                "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "reserved_memory_mb": torch.cuda.memory_reserved() / (1024**2),
                "samples": self.memory_samples,
            }

            # Add detailed CUDA memory stats if available
            if hasattr(torch.cuda, "memory_stats"):
                cuda_stats = torch.cuda.memory_stats()
                base_stats.update(
                    {
                        "allocated_bytes_all_peak_mb": cuda_stats.get(
                            "allocated_bytes.all.peak", 0
                        )
                        / (1024**2),
                        "reserved_bytes_all_peak_mb": cuda_stats.get(
                            "reserved_bytes.all.peak", 0
                        )
                        / (1024**2),
                    }
                )

            return base_stats
        elif self.device == "mps":
            current_memory = self._estimate_mps_memory()
            stats = {
                "device": "mps",
                "current_memory_mb": current_memory,
                "peak_memory_mb": self.peak_memory_mb,
                "reserved_memory_mb": current_memory,  # Same as current for MPS
                "samples": self.memory_samples,
            }

            # Add driver_allocated_memory if available
            if hasattr(torch.mps, "driver_allocated_memory"):
                stats["driver_allocated_memory_mb"] = (
                    torch.mps.driver_allocated_memory() / (1024**2)
                )

            return stats
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
