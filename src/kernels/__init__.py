"""
CUDA/Triton kernels for speculative decoding with safe fallbacks.
"""

import logging
from typing import Any, Callable, Dict, Optional

import torch

from .build import load_kernels
from .reference import kv_append_ref, verify_prefix_ref
from .registry import registry

logger = logging.getLogger(__name__)

# Load available kernels
_kernels = load_kernels()


# Register all available backends
def _register_kernels():
    """Register all available kernel backends."""

    # Register CUDA kernels if available
    if _kernels.get("verify_prefix") and hasattr(_kernels["verify_prefix"], "__name__"):
        if "cuda" in _kernels["verify_prefix"].__name__.lower():
            registry.register(
                "verify_prefix", _kernels["verify_prefix"], priority=100, device="cuda"
            )

    if _kernels.get("kv_append") and hasattr(_kernels["kv_append"], "__name__"):
        if "cuda" in _kernels["kv_append"].__name__.lower():
            registry.register(
                "kv_append", _kernels["kv_append"], priority=100, device="cuda"
            )

    # Register Triton kernels if available
    try:
        from .triton.verify import verify_prefix_triton

        registry.register(
            "verify_prefix", verify_prefix_triton, priority=50, device="cuda"
        )
        registry.register(
            "verify_prefix", verify_prefix_triton, priority=50, device="mps"
        )
    except ImportError:
        pass

    # Register reference implementations (fallback)
    registry.register("verify_prefix", verify_prefix_ref, priority=10, device="auto")
    registry.register("kv_append", kv_append_ref, priority=10, device="auto")


# Register kernels
_register_kernels()


# Export kernel functions using registry
def get_verify_prefix(device: Optional[str] = None):
    """Get verify_prefix kernel for device."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return registry.get_best("verify_prefix", device)


def get_kv_append(device: Optional[str] = None):
    """Get kv_append kernel for device."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return registry.get_best("kv_append", device)


# Backward compatibility
verify_prefix = get_verify_prefix()
kv_append = get_kv_append()


# Export backend information
def get_kernel_info():
    """Get information about loaded kernels."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    status = registry.get_status(device)

    return {
        "verify_backend": status.get("verify_prefix", "unknown"),
        "kv_append_backend": status.get("kv_append", "unknown"),
        "verify_available": verify_prefix is not None,
        "kv_append_available": kv_append is not None,
        "device": device,
    }


def log_kernel_status():
    """Log which kernel backends are being used."""
    info = get_kernel_info()
    logger.info(
        f"Kernel backends: verify={info['verify_backend']}, kv_append={info['kv_append_backend']}"
    )

    if not info["verify_available"]:
        logger.warning("verify_prefix kernel not available, using fallback")
    if not info["kv_append_available"]:
        logger.warning("kv_append kernel not available, using fallback")


# Log kernel status on import
log_kernel_status()

__all__ = [
    "verify_prefix",
    "kv_append",
    "get_verify_prefix",
    "get_kv_append",
    "get_kernel_info",
    "log_kernel_status",
    "registry",
]
