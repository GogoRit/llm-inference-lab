"""
Scheduler Module - Request Batching & Routing

This module handles intelligent request batching, scheduling, and routing
to optimize GPU utilization and reduce latency for LLM inference.

Components:
- Dynamic batching algorithms
- Request queue management
- Load balancing across multiple GPUs
- Priority-based scheduling
- Batch size optimization
- Request timeout handling
- Resource allocation and monitoring
- Speculative decoding scheduling
"""

from .speculative_scheduler import SpeculativeScheduler, create_speculative_scheduler

__all__ = [
    "SpeculativeScheduler",
    "create_speculative_scheduler",
]
