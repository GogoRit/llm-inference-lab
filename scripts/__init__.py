"""
Scripts Module

This module contains utility scripts for benchmarking, testing, and development.

Scripts:
- comprehensive_k_sweep.py: Main K-sweep benchmarking script
- benchmarks/sequence_pool_benchmark.py: Sequence pool performance tests
- dev/smoke_cuda.py: CUDA smoke tests
- k_sweep/: K-sweep utilities (runner, plotting, results, utils)
- microbench_verify.py: Micro-benchmark for verification
"""

import sys
from pathlib import Path

# Script utilities and helpers

# Add src to Python path for script execution
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
