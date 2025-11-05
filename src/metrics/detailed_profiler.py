"""
Detailed profiler for kernel timings and performance metrics.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class DetailedProfiler:
    """Profiler for detailed kernel and performance metrics."""

    def __init__(self, enabled: Optional[bool] = None):
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is enabled. If None, checks SPECDEC_DETAILED_METRICS env var.
        """
        if enabled is None:
            enabled = os.getenv("SPECDEC_DETAILED_METRICS", "0") == "1"

        self.enabled = enabled
        self.metrics: Dict[str, Any] = {
            "kernel_times": defaultdict(list),
            "acceptance_histogram": defaultdict(int),
            "gpu_memory_peak": 0,
            "gpu_memory_samples": [],
            "step_times": [],
            "total_steps": 0,
        }

        if self.enabled:
            logger.info("Detailed profiling enabled")
        else:
            logger.debug("Detailed profiling disabled")

    def record_kernel_time(self, name: str, time_ms: float) -> None:
        """Record kernel execution time."""
        if not self.enabled:
            return

        kernel_times = self.metrics["kernel_times"]
        if isinstance(kernel_times, defaultdict):
            kernel_times[name].append(time_ms)
        logger.debug(f"Kernel {name}: {time_ms:.3f}ms")

    def record_acceptance(self, accepted_len: int, total_len: int) -> None:
        """Record acceptance pattern."""
        if not self.enabled:
            return

        acceptance_hist = self.metrics["acceptance_histogram"]
        if isinstance(acceptance_hist, defaultdict):
            acceptance_hist[accepted_len] += 1
        logger.debug(f"Accepted {accepted_len}/{total_len} tokens")

    def record_gpu_memory(self) -> None:
        """Record current GPU memory usage."""
        if not self.enabled or not torch.cuda.is_available():
            return

        current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_samples = self.metrics["gpu_memory_samples"]
        if isinstance(gpu_samples, list):
            gpu_samples.append(current_memory)
        peak_memory = self.metrics["gpu_memory_peak"]
        if isinstance(peak_memory, (int, float)):
            self.metrics["gpu_memory_peak"] = max(peak_memory, current_memory)

    def record_step_time(self, step_time_ms: float) -> None:
        """Record total step time."""
        if not self.enabled:
            return

        step_times = self.metrics["step_times"]
        if isinstance(step_times, list):
            step_times.append(step_time_ms)
        total_steps = self.metrics["total_steps"]
        if isinstance(total_steps, int):
            self.metrics["total_steps"] = total_steps + 1

    def reset_peak_memory(self) -> None:
        """Reset peak memory stats (call before new run)."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.metrics["gpu_memory_peak"] = 0
            self.metrics["gpu_memory_samples"] = []

    def summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics."""
        if not self.enabled:
            return {"enabled": False}

        summary = {
            "enabled": True,
            "total_steps": self.metrics["total_steps"],
            "kernel_times_avg": {},
            "acceptance_histogram": dict(self.metrics["acceptance_histogram"]),
            "gpu_memory_peak_mb": self.metrics["gpu_memory_peak"],
            "gpu_memory_avg_mb": 0,
            "step_times_avg_ms": 0,
        }

        # Calculate average kernel times
        kernel_times = self.metrics["kernel_times"]
        if isinstance(kernel_times, defaultdict):
            for name, times in kernel_times.items():
                if isinstance(times, list) and times:
                    summary["kernel_times_avg"][name] = sum(times) / len(times)

        # Calculate average GPU memory
        gpu_samples = self.metrics["gpu_memory_samples"]
        if isinstance(gpu_samples, list) and gpu_samples:
            summary["gpu_memory_avg_mb"] = sum(gpu_samples) / len(gpu_samples)

        # Calculate average step time
        step_times = self.metrics["step_times"]
        if isinstance(step_times, list) and step_times:
            summary["step_times_avg_ms"] = sum(step_times) / len(step_times)

        return summary

    def get_kernel_timing_context(self, name: str):
        """Get context manager for timing kernel execution."""
        if not self.enabled:
            return _NoopContext()

        return _KernelTimingContext(self, name)


class _NoopContext:
    """No-op context manager for when profiling is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _KernelTimingContext:
    """Context manager for timing kernel execution."""

    def __init__(self, profiler: DetailedProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if not self.profiler.enabled:
            return self

        if torch.cuda.is_available():
            # Use CUDA events for accurate timing
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            # Use CPU timing
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, *args):
        if not self.profiler.enabled:
            return

        if torch.cuda.is_available() and self.start_event is not None:
            # Use CUDA events
            self.end_event.record()
            torch.cuda.synchronize()
            time_ms = self.start_event.elapsed_time(self.end_event)
        elif self.start_time is not None:
            # Use CPU timing
            time_ms = (time.perf_counter() - self.start_time) * 1000
        else:
            return

        self.profiler.record_kernel_time(self.name, time_ms)


# Global profiler instance
_profiler = None


def get_profiler() -> DetailedProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = DetailedProfiler()
    return _profiler


def record_kernel_time(name: str, time_ms: float) -> None:
    """Record kernel time using global profiler."""
    get_profiler().record_kernel_time(name, time_ms)


def record_acceptance(accepted_len: int, total_len: int) -> None:
    """Record acceptance pattern using global profiler."""
    get_profiler().record_acceptance(accepted_len, total_len)


def record_gpu_memory() -> None:
    """Record GPU memory using global profiler."""
    get_profiler().record_gpu_memory()


def get_kernel_timing_context(name: str):
    """Get kernel timing context using global profiler."""
    return get_profiler().get_kernel_timing_context(name)


def get_summary() -> Dict[str, Any]:
    """Get profiler summary."""
    return get_profiler().summary()
