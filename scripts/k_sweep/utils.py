"""
Utility functions for K-sweep benchmarking.

Provides system information gathering, device resolution, and deterministic mode setup.
"""

import logging
import os
import platform
from datetime import datetime
from typing import Dict

import torch

from kernels import get_kernel_info

logger = logging.getLogger(__name__)


def get_system_info(device: str) -> Dict:
    """Get system and environment metadata."""
    # Get kernel backend info
    kinfo = get_kernel_info()

    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available()
            else (
                "MPS"
                if device == "mps" and torch.backends.mps.is_available()
                else "CPU"
            )
        ),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "dtype": "float16" if device in ["cuda", "mps"] else "float32",
        "kernel_backends": {
            "verify": kinfo.get("verify_backend", "unknown"),
            "kv_append": kinfo.get("kv_append_backend", "unknown"),
        },
        "kv_append_enabled": os.getenv("SPECDEC_ENABLE_KV_APPEND", "1").lower()
        in ("1", "true", "yes"),
        "batch_size": int(os.getenv("SPECDEC_BATCH_SIZE", "8")),
        "parallel_streams": os.getenv("SPECDEC_PARALLEL_STREAMS", "1").lower()
        in ("1", "true", "yes"),
        "cuda_graph": os.getenv("SPECDEC_CUDA_GRAPH", "0").lower()
        in ("1", "true", "yes"),
    }

    # Add GPU memory info if CUDA
    if device == "cuda" and torch.cuda.is_available():
        info["cuda_total_memory_gb"] = torch.cuda.get_device_properties(
            0
        ).total_memory / (1024**3)
        info["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        info["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

    return info


def resolve_device(device_arg: str) -> str:
    """Resolve device argument to actual device. MPS-first for testing."""
    if device_arg == "auto":
        # Test MPS first, then CUDA (MPS-first approach for development)
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device_arg


def set_deterministic_mode(enable: bool) -> None:
    """Enable reproducible behavior across libraries."""
    if not enable:
        return
    try:
        import random

        import numpy as np
        import torch

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.warning(f"Failed to set deterministic mode: {e}")
