"""
Kernel registry for safe backend selection with priority and device filtering.
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class KernelRegistry:
    """Registry for kernel implementations with priority-based selection."""

    _kernels: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls, op_name: str, impl: Callable, priority: int, device: str = "auto"
    ) -> None:
        """
        Register a kernel implementation.

        Args:
            op_name: Operation name (e.g., 'verify_prefix', 'kv_append')
            impl: Kernel function
            priority: Higher priority = preferred (0-100)
            device: Target device ('cuda', 'mps', 'cpu', 'auto')
        """
        if op_name not in cls._kernels:
            cls._kernels[op_name] = []

        cls._kernels[op_name].append(
            {"function": impl, "priority": priority, "device": device}
        )

        # Sort by priority (highest first)
        cls._kernels[op_name].sort(key=lambda x: x["priority"], reverse=True)

        logger.debug(
            f"Registered {op_name} kernel: {impl.__name__} (priority={priority}, device={device})"
        )

    @classmethod
    def get_best(cls, op_name: str, device: str) -> Optional[Callable]:
        """
        Get the best available kernel for operation and device.

        Args:
            op_name: Operation name
            device: Target device

        Returns:
            Best available kernel function or None
        """
        if op_name not in cls._kernels:
            return None

        # Filter by device compatibility
        candidates = []
        for kernel in cls._kernels[op_name]:
            if kernel["device"] in [device, "auto"]:
                candidates.append(kernel)

        if not candidates:
            logger.warning(f"No kernels available for {op_name} on device {device}")
            return None

        # Return highest priority candidate
        best = candidates[0]
        logger.debug(
            f"Selected {op_name} kernel: {best['function'].__name__} (priority={best['priority']}, device={best['device']})"
        )
        return best["function"]

    @classmethod
    def list_available(cls, op_name: str, device: str) -> list:
        """List all available kernels for operation and device."""
        if op_name not in cls._kernels:
            return []

        return [
            {
                "name": kernel["function"].__name__,
                "priority": kernel["priority"],
                "device": kernel["device"],
            }
            for kernel in cls._kernels[op_name]
            if kernel["device"] in [device, "auto"]
        ]

    @classmethod
    def get_status(cls, device: str) -> Dict[str, str]:
        """Get status of all registered kernels for device."""
        status = {}
        for op_name in cls._kernels:
            best = cls.get_best(op_name, device)
            if best:
                status[op_name] = f"{best.__name__}"
            else:
                status[op_name] = "none"
        return status


# Global registry instance
registry = KernelRegistry()
