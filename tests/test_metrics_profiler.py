"""
Tests for metrics profiler.
"""

# import pytest  # Unused import
# import torch  # Unused import

from src.metrics.detailed_profiler import DetailedProfiler


class TestDetailedProfiler:
    """Test detailed profiler functionality."""

    def test_profiler_disabled_by_default(self):
        """Test that profiler is disabled by default."""
        profiler = DetailedProfiler(enabled=False)
        assert not profiler.enabled

        # Should not record anything
        profiler.record_kernel_time("test", 1.0)
        profiler.record_acceptance(2, 4)
        profiler.record_gpu_memory()

        summary = profiler.summary()
        assert not summary["enabled"]

    def test_profiler_enabled(self):
        """Test profiler when enabled."""
        profiler = DetailedProfiler(enabled=True)
        assert profiler.enabled

        # Record some metrics
        profiler.record_kernel_time("test_kernel", 10.0)
        profiler.record_kernel_time("test_kernel", 20.0)
        profiler.record_acceptance(2, 4)
        profiler.record_acceptance(1, 4)
        profiler.record_step_time(100.0)

        summary = profiler.summary()
        assert summary["enabled"]
        assert summary["total_steps"] == 1
        assert summary["kernel_times_avg"]["test_kernel"] == 15.0
        assert summary["acceptance_histogram"][2] == 1
        assert summary["acceptance_histogram"][1] == 1
        assert summary["step_times_avg_ms"] == 100.0

    def test_kernel_timing_context(self):
        """Test kernel timing context manager."""
        profiler = DetailedProfiler(enabled=True)

        # Test with context manager
        with profiler.get_kernel_timing_context(
            "test_kernel"
        ):  # as ctx:  # Unused variable
            # Simulate some work
            import time

            time.sleep(0.001)  # 1ms

        summary = profiler.summary()
        assert "test_kernel" in summary["kernel_times_avg"]
        assert summary["kernel_times_avg"]["test_kernel"] > 0

    def test_gpu_memory_recording(self):
        """Test GPU memory recording."""
        profiler = DetailedProfiler(enabled=True)

        # Reset peak memory
        profiler.reset_peak_memory()

        # Record memory
        profiler.record_gpu_memory()

        summary = profiler.summary()
        assert "gpu_memory_peak_mb" in summary
        assert "gpu_memory_avg_mb" in summary

    def test_acceptance_histogram(self):
        """Test acceptance histogram recording."""
        profiler = DetailedProfiler(enabled=True)

        # Record various acceptance patterns
        profiler.record_acceptance(0, 4)
        profiler.record_acceptance(1, 4)
        profiler.record_acceptance(2, 4)
        profiler.record_acceptance(2, 4)
        profiler.record_acceptance(4, 4)

        summary = profiler.summary()
        hist = summary["acceptance_histogram"]

        assert hist[0] == 1
        assert hist[1] == 1
        assert hist[2] == 2
        assert hist[4] == 1
        assert 3 not in hist

    def test_step_timing(self):
        """Test step timing recording."""
        profiler = DetailedProfiler(enabled=True)

        # Record step times
        profiler.record_step_time(100.0)
        profiler.record_step_time(200.0)
        profiler.record_step_time(150.0)

        summary = profiler.summary()
        assert summary["total_steps"] == 3
        assert summary["step_times_avg_ms"] == 150.0

    def test_empty_summary(self):
        """Test summary when no data recorded."""
        profiler = DetailedProfiler(enabled=True)

        summary = profiler.summary()
        assert summary["enabled"]
        assert summary["total_steps"] == 0
        assert summary["kernel_times_avg"] == {}
        assert summary["acceptance_histogram"] == {}
        assert summary["gpu_memory_peak_mb"] == 0
        assert summary["gpu_memory_avg_mb"] == 0
        assert summary["step_times_avg_ms"] == 0
