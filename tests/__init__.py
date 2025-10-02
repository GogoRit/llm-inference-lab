"""
Test Suite for LLM Inference Lab

This package contains comprehensive tests for all modules including unit tests,
integration tests, and performance tests.

Test Categories:
- Unit tests for individual components
- Integration tests for module interactions
- Performance tests and benchmarks
- GPU-specific tests (marked with @pytest.mark.gpu)
- Mock tests for CI/CD environments
"""

# Test configuration and utilities
import pytest
import torch

# Note: GPU tests are marked with @pytest.mark.gpu and will be skipped
# automatically if CUDA is not available. CPU/MPS tests should always run.
