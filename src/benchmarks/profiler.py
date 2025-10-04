"""
Performance Profiling Utilities for LLM Inference Lab

Provides comprehensive profiling capabilities including PyTorch Profiler integration,
memory tracking, and performance analysis tools for CPU/MPS optimization.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from torch.profiler import ProfilerActivity, profile, record_function

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiler for LLM inference optimization."""

    def __init__(
        self,
        enable_profiling: bool = True,
        profile_dir: Optional[str] = None,
        memory_tracking: bool = True,
        device: str = "auto",
    ):
        """
        Initialize the performance profiler.

        Args:
            enable_profiling: Whether to enable PyTorch profiling
            profile_dir: Directory to save profiling traces
            memory_tracking: Whether to track memory usage
            device: Device to profile on
        """
        self.enable_profiling = enable_profiling
        self.memory_tracking = memory_tracking
        self.device = self._select_device(device)

        # Setup profiling directory
        if profile_dir:
            self.profile_dir = Path(profile_dir)
            self.profile_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.profile_dir = Path("profiles")
            self.profile_dir.mkdir(exist_ok=True)

        # Memory tracking
        self.process = psutil.Process()
        self.memory_samples: List[Dict[str, float]] = []
        self.peak_memory_mb = 0.0

        # Performance metrics
        self.timing_data: Dict[str, List[float]] = {}
        self.profiler: Optional[Any] = None

        logger.info(
            f"PerformanceProfiler initialized: device={self.device}, "
            f"profiling={enable_profiling}, memory_tracking={memory_tracking}"
        )

    def _select_device(self, device: str) -> str:
        """Select the best available device for profiling."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    @contextmanager
    def profile_context(self, name: str, save_trace: bool = True):
        """
        Context manager for profiling specific operations.

        Args:
            name: Name of the operation being profiled
            save_trace: Whether to save the profiling trace to file
        """
        if not self.enable_profiling:
            yield
            return

        # Determine activities based on device
        activities = [ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        elif self.device == "mps":
            # MPS doesn't have specific profiler activity, use CPU
            activities = [ProfilerActivity.CPU]

        # Create profiler
        self.profiler = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )

        try:
            self.profiler.start()
            yield
        finally:
            self.profiler.stop()

            if save_trace:
                self._save_trace(name)

    def _save_trace(self, name: str) -> None:
        """Save profiling trace to file."""
        if not self.profiler:
            return

        trace_file = self.profile_dir / f"{name}_trace.json"
        self.profiler.export_chrome_trace(str(trace_file))
        logger.info(f"Profiling trace saved to {trace_file}")

    def start_memory_tracking(self) -> None:
        """Start memory usage tracking."""
        if not self.memory_tracking:
            return

        self.memory_samples = []
        self.peak_memory_mb = 0.0
        self._sample_memory("start")

    def stop_memory_tracking(self) -> Dict[str, float]:
        """
        Stop memory tracking and return summary statistics.

        Returns:
            Dictionary containing memory usage statistics
        """
        if not self.memory_tracking or not self.memory_samples:
            return {}

        self._sample_memory("end")

        # Calculate statistics
        memory_values = [sample["rss_mb"] for sample in self.memory_samples]

        stats = {
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "min_memory_mb": min(memory_values),
            "max_memory_mb": max(memory_values),
            "memory_samples": len(self.memory_samples),
        }

        # Add device-specific memory if available
        if self.device == "cuda" and torch.cuda.is_available():
            stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        elif self.device == "mps" and torch.backends.mps.is_available():
            # MPS doesn't have direct memory tracking, use process memory
            stats["mps_estimated_mb"] = stats["peak_memory_mb"]

        logger.info(f"Memory tracking completed: {stats}")
        return stats

    def _sample_memory(self, phase: str) -> None:
        """Sample current memory usage."""
        try:
            memory_info = self.process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024

            sample = {
                "phase": phase,
                "timestamp": time.time(),
                "rss_mb": rss_mb,
            }

            self.memory_samples.append(sample)
            self.peak_memory_mb = max(self.peak_memory_mb, rss_mb)

        except Exception as e:
            logger.warning(f"Failed to sample memory: {e}")

    def time_operation(self, name: str, operation, *args, **kwargs):
        """
        Time a specific operation and record the results.

        Args:
            name: Name of the operation
            operation: Callable to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation
        """
        start_time = time.time()

        try:
            result = operation(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            if name not in self.timing_data:
                self.timing_data[name] = []
            self.timing_data[name].append(duration_ms)

            logger.debug(f"Operation '{name}' took {duration_ms:.2f}ms")

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics for all recorded operations.

        Returns:
            Dictionary containing timing statistics for each operation
        """
        stats = {}

        for name, times in self.timing_data.items():
            if not times:
                continue

            stats[name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }

            if len(times) > 1:
                # Calculate standard deviation
                mean = stats[name]["avg_ms"]
                variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
                stats[name]["std_ms"] = variance**0.5
            else:
                stats[name]["std_ms"] = 0.0

        return stats

    def profile_model_forward(
        self, model: torch.nn.Module, input_ids: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        """
        Profile a model forward pass with comprehensive metrics.

        Args:
            model: PyTorch model to profile
            input_ids: Input tensor
            **kwargs: Additional arguments for model forward

        Returns:
            Dictionary containing profiling results
        """
        results: Dict[str, Any] = {}

        # Memory tracking
        if self.memory_tracking:
            self.start_memory_tracking()

        # Time the forward pass
        with self.profile_context("model_forward"):
            start_time = time.time()

            with record_function("model_forward"):
                outputs = model(input_ids, **kwargs)

            end_time = time.time()
            forward_time_ms = (end_time - start_time) * 1000

        # Memory statistics
        if self.memory_tracking:
            memory_stats = self.stop_memory_tracking()
            results.update(memory_stats)

        # Timing statistics
        results.update(
            {
                "forward_time_ms": forward_time_ms,
                "input_tokens": float(input_ids.shape[1]),
            }
        )

        # Metadata (separate from numeric results)
        results["device"] = str(input_ids.device)
        results["dtype"] = str(input_ids.dtype)

        # Add model-specific metrics
        if hasattr(outputs, "logits"):
            results["output_vocab_size"] = outputs.logits.shape[-1]
            results["batch_size"] = outputs.logits.shape[0]

        return results

    def profile_generation(
        self, generate_func, prompt: str, max_tokens: int, **kwargs
    ) -> Dict[str, Any]:
        """
        Profile text generation with comprehensive metrics.

        Args:
            generate_func: Function that performs text generation
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for generation

        Returns:
            Dictionary containing generation profiling results
        """
        results = {}

        # Memory tracking
        if self.memory_tracking:
            self.start_memory_tracking()

        # Time the generation
        with self.profile_context("text_generation"):
            start_time = time.time()

            with record_function("text_generation"):
                generation_result = generate_func(
                    prompt, max_tokens=max_tokens, **kwargs
                )

            end_time = time.time()
            generation_time_ms = (end_time - start_time) * 1000

        # Memory statistics
        if self.memory_tracking:
            memory_stats = self.stop_memory_tracking()
            results.update(memory_stats)

        # Extract generation metrics
        if isinstance(generation_result, dict):
            results.update(
                {
                    "generation_time_ms": generation_time_ms,
                    "prompt_length": len(prompt.split()),
                    "generated_tokens": generation_result.get("generated_tokens", []),
                    "generated_text": generation_result.get("text", ""),
                    "latency_ms": generation_result.get(
                        "latency_ms", generation_time_ms
                    ),
                    "tokens_per_sec": generation_result.get("tokens_per_sec", 0.0),
                }
            )

            # Add speculative decoding specific metrics
            if "acceptance_rate" in generation_result:
                results["acceptance_rate"] = generation_result["acceptance_rate"]
            if "proposed" in generation_result:
                results["proposed_tokens"] = generation_result["proposed"]
            if "accepted" in generation_result:
                results["accepted_tokens"] = generation_result["accepted"]

        return results

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Returns:
            Dictionary containing all performance metrics
        """
        report = {
            "device": self.device,
            "profiling_enabled": self.enable_profiling,
            "memory_tracking_enabled": self.memory_tracking,
            "timing_stats": self.get_timing_stats(),
        }

        # Add memory statistics if available
        if self.memory_samples:
            memory_values = [sample["rss_mb"] for sample in self.memory_samples]
            report["memory_stats"] = {
                "peak_memory_mb": max(memory_values),
                "avg_memory_mb": sum(memory_values) / len(memory_values),
                "min_memory_mb": min(memory_values),
                "max_memory_mb": max(memory_values),
                "samples": len(self.memory_samples),
            }

        return report

    def reset(self) -> None:
        """Reset all profiling data."""
        self.memory_samples = []
        self.peak_memory_mb = 0.0
        self.timing_data = {}
        self.profiler = None
        logger.info("Profiler reset")


def create_profiler(
    enable_profiling: bool = True,
    profile_dir: Optional[str] = None,
    memory_tracking: bool = True,
    device: str = "auto",
) -> PerformanceProfiler:
    """
    Create a PerformanceProfiler instance.

    Args:
        enable_profiling: Whether to enable PyTorch profiling
        profile_dir: Directory to save profiling traces
        memory_tracking: Whether to track memory usage
        device: Device to profile on

    Returns:
        Configured PerformanceProfiler instance
    """
    return PerformanceProfiler(
        enable_profiling=enable_profiling,
        profile_dir=profile_dir,
        memory_tracking=memory_tracking,
        device=device,
    )


# Convenience functions for common profiling tasks
@contextmanager
def profile_operation(name: str, profiler: Optional[PerformanceProfiler] = None):
    """Context manager for profiling a single operation."""
    if profiler is None:
        profiler = create_profiler()

    with profiler.profile_context(name):
        yield profiler


def time_function(
    name: str, func, *args, profiler: Optional[PerformanceProfiler] = None, **kwargs
):
    """Time a function execution."""
    if profiler is None:
        profiler = create_profiler()

    return profiler.time_operation(name, func, *args, **kwargs)
