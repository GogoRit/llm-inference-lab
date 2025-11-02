"""
Deterministic seeding and reproducibility utilities for Phase 3D.

Provides helper functions to set deterministic mode for reproducible
benchmarks across CUDA, MPS, and CPU backends.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_deterministic_mode(seed: Optional[int] = None, device: str = "auto") -> None:
    """
    Set deterministic mode for reproducible benchmarks.

    Args:
        seed: Random seed (default: 1234 if None)
        device: Target device (auto, cuda, mps, cpu)

    This function:
    - Sets random seeds for Python, NumPy, and PyTorch
    - Configures CuDNN deterministic mode (if CUDA available)
    - Disables CuDNN benchmarking for reproducibility
    """
    if seed is None:
        seed = 1234

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set device-specific seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure CuDNN for deterministic behavior (CUDA only)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_deterministic() -> None:
    """
    Check environment flag and set deterministic mode if requested.

    Reads SPECDEC_DETERMINISTIC environment variable.
    If set to "1", "true", or "yes", calls set_deterministic_mode().
    """
    env_value = os.getenv("SPECDEC_DETERMINISTIC", "0").lower()
    if env_value in ("1", "true", "yes"):
        set_deterministic_mode()
