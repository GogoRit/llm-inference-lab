"""
Tests for kernel registry.
"""

# import pytest  # Unused import
# import torch  # Unused import

from src.kernels.registry import KernelRegistry


class TestKernelRegistry:
    """Test kernel registry functionality."""

    def test_register_and_get_best(self):
        """Test kernel registration and selection."""
        # Clear registry
        KernelRegistry._kernels.clear()

        # Mock kernel functions
        def cuda_kernel():
            return "cuda"

        def triton_kernel():
            return "triton"

        def torch_kernel():
            return "torch"

        # Register kernels
        KernelRegistry.register("test_op", cuda_kernel, priority=100, device="cuda")
        KernelRegistry.register("test_op", triton_kernel, priority=50, device="cuda")
        KernelRegistry.register("test_op", torch_kernel, priority=10, device="auto")

        # Test selection
        best_cuda = KernelRegistry.get_best("test_op", "cuda")
        assert best_cuda == cuda_kernel

        best_cpu = KernelRegistry.get_best("test_op", "cpu")
        assert best_cpu == torch_kernel

        # Test unknown operation
        unknown = KernelRegistry.get_best("unknown_op", "cuda")
        assert unknown is None

    def test_priority_ordering(self):
        """Test that kernels are ordered by priority."""
        # Clear registry
        KernelRegistry._kernels.clear()

        def low_priority():
            return "low"

        def high_priority():
            return "high"

        # Register in reverse order
        KernelRegistry.register("test_op", low_priority, priority=10, device="cuda")
        KernelRegistry.register("test_op", high_priority, priority=100, device="cuda")

        # Should get highest priority
        best = KernelRegistry.get_best("test_op", "cuda")
        assert best == high_priority

    def test_device_filtering(self):
        """Test device-based filtering."""
        # Clear registry
        KernelRegistry._kernels.clear()

        def cuda_kernel():
            return "cuda"

        def auto_kernel():
            return "auto"

        # Register kernels
        KernelRegistry.register("test_op", cuda_kernel, priority=100, device="cuda")
        KernelRegistry.register("test_op", auto_kernel, priority=50, device="auto")

        # Test device filtering
        cuda_result = KernelRegistry.get_best("test_op", "cuda")
        assert cuda_result == cuda_kernel

        cpu_result = KernelRegistry.get_best("test_op", "cpu")
        assert cpu_result == auto_kernel

        # Test device not available
        mps_result = KernelRegistry.get_best("test_op", "mps")
        assert mps_result == auto_kernel  # auto should work for mps

    def test_list_available(self):
        """Test listing available kernels."""
        # Clear registry
        KernelRegistry._kernels.clear()

        def cuda_kernel():
            return "cuda"

        def auto_kernel():
            return "auto"

        # Register kernels
        KernelRegistry.register("test_op", cuda_kernel, priority=100, device="cuda")
        KernelRegistry.register("test_op", auto_kernel, priority=50, device="auto")

        # Test listing
        available = KernelRegistry.list_available("test_op", "cuda")
        assert len(available) == 2
        assert available[0]["name"] == "cuda_kernel"
        assert available[0]["priority"] == 100

        # Test unknown operation
        unknown = KernelRegistry.list_available("unknown_op", "cuda")
        assert len(unknown) == 0

    def test_get_status(self):
        """Test getting status of all kernels."""
        # Clear registry
        KernelRegistry._kernels.clear()

        def cuda_kernel():
            return "cuda"

        def auto_kernel():
            return "auto"

        # Register kernels
        KernelRegistry.register("test_op", cuda_kernel, priority=100, device="cuda")
        KernelRegistry.register("test_op", auto_kernel, priority=50, device="auto")

        # Test status
        status = KernelRegistry.get_status("cuda")
        assert "test_op" in status
        assert status["test_op"] == "cuda_kernel"

        # Test device with no kernels
        status_cpu = KernelRegistry.get_status("cpu")
        assert "test_op" in status_cpu
        assert status_cpu["test_op"] == "auto_kernel"
