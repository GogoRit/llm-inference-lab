"""
Speculative Decoding Package for LLM Inference Lab

This package implements speculative decoding techniques for faster LLM inference
using a draft model to propose tokens and a base model to verify them.

Key Components:
- draft_model: Small draft model for token proposal
- verifier: Base model verification with exact-match policy
- pipeline: Orchestrates the speculative decoding loop
- run_specdec: CLI entrypoint for speculative decoding
"""

from .draft_model import DraftModel
from .pipeline import SpeculativePipeline
from .verifier import Verifier

__all__ = ["DraftModel", "Verifier", "SpeculativePipeline"]
