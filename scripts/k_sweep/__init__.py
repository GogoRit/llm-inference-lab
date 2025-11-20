"""
K-Sweep Benchmarking Module

Organized module for comprehensive K-sweep benchmarking of speculative decoding.
All scripts use a unified result format for consistency.
"""

from .plotting import create_plots
from .results import save_results
from .runner import PROMPT_SUITE, run_comprehensive_k_sweep
from .utils import get_system_info, resolve_device, set_deterministic_mode

__all__ = [
    "PROMPT_SUITE",
    "get_system_info",
    "resolve_device",
    "set_deterministic_mode",
    "run_comprehensive_k_sweep",
    "save_results",
    "create_plots",
]
