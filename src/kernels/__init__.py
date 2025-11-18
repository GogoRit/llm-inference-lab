"""
CUDA/Triton kernels for speculative decoding with safe fallbacks.
"""

import logging
import os
from typing import Optional

import torch

from .build import load_kernels
from .reference import kv_append_ref, verify_prefix_ref
from .registry import registry

logger = logging.getLogger(__name__)

# Check if we should force PyTorch backend (skip Triton)
FORCE_PYTORCH_BACKEND = os.getenv("SPECDEC_FORCE_PYTORCH_BACKEND", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Load available kernels
_kernels = load_kernels()

# Track if we've logged a Triton fallback warning (one per process)
_triton_fallback_warned = False


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

    # Register Triton kernels if available and not forced to PyTorch
    if not FORCE_PYTORCH_BACKEND:
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
        except Exception as e:
            global _triton_fallback_warned
            if not _triton_fallback_warned:
                logger.warning(
                    f"Triton verify kernel failed to load (falling back to PyTorch): {e}"
                )
                _triton_fallback_warned = True
    else:
        logger.info(
            "SPECDEC_FORCE_PYTORCH_BACKEND is set, skipping Triton verify kernel registration"
        )

    # Register reference implementations (fallback)
    registry.register("verify_prefix", verify_prefix_ref, priority=10, device="auto")
    # Register simplified kv_append (simple concat version)
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

    # Map function names to readable backend names
    verify_func_name = status.get("verify_prefix", "unknown")
    if verify_func_name == "unknown":
        verify_backend = "unknown"
    elif (
        "cuda" in verify_func_name.lower() and "triton" not in verify_func_name.lower()
    ):
        verify_backend = "cuda"
    elif "triton" in verify_func_name.lower():
        verify_backend = "triton"
    elif "ref" in verify_func_name.lower():
        verify_backend = "torch"
    else:
        verify_backend = verify_func_name

    kv_func_name = status.get("kv_append", "unknown")
    if kv_func_name == "unknown":
        kv_backend = "unknown"
    elif "cuda" in kv_func_name.lower():
        kv_backend = "cuda"
    elif "ref" in kv_func_name.lower():
        kv_backend = "torch"
    else:
        kv_backend = kv_func_name

    return {
        "verify_backend": verify_backend,
        "kv_append_backend": kv_backend,
        "verify_available": verify_prefix is not None,
        "kv_append_available": kv_append is not None,
        "device": device,
    }


def log_kernel_status():
    """Log which kernel backends are being used."""
    info = get_kernel_info()
    logger.info(f"Using verify backend: {info['verify_backend']}")
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
