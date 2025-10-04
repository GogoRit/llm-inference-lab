"""
Optimization Package for LLM Inference Lab

Provides CPU/MPS optimized utilities for improved performance and memory efficiency.
"""

from .mixed_precision import (
    GradientCheckpointingManager,
    LocalOptimizationManager,
    MixedPrecisionManager,
    create_optimization_manager,
)
from .tokenizer_optimization import (
    OptimizedTokenizer,
    create_optimized_tokenizer,
)

__all__ = [
    "LocalOptimizationManager",
    "MixedPrecisionManager",
    "GradientCheckpointingManager",
    "create_optimization_manager",
    "OptimizedTokenizer",
    "create_optimized_tokenizer",
]
