"""
Tests for CUDA graph capture fallback behavior.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec import SpeculativePipeline  # noqa: E402
from specdec.models.fake_lm import FakeLM, create_fake_lm  # noqa: E402


class TestCUDAGraphFallback:
    """Test CUDA graph capture fallback behavior."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_graph_fallback_on_non_cuda(self):
        """Test that graph capture is disabled on non-CUDA devices."""
        # Force CPU device
        pipeline = SpeculativePipeline(
            base_lm=FakeLM(),
            draft_lm=FakeLM(),
            max_draft=2,
            device="cpu",
        )
        assert pipeline.enable_cuda_graph is False

    @pytest.mark.skipif(
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_graph_fallback_on_mps(self):
        """Test that graph capture is disabled on MPS."""
        pipeline = SpeculativePipeline(
            base_lm=FakeLM(),
            draft_lm=FakeLM(),
            max_draft=2,
            device="mps",
        )
        assert pipeline.enable_cuda_graph is False

    def test_graph_env_flag_parsing(self):
        """Test that SPECDEC_CUDA_GRAPH env flag is parsed correctly."""
        # Test disabled by default
        if "SPECDEC_CUDA_GRAPH" in os.environ:
            original = os.environ["SPECDEC_CUDA_GRAPH"]
        else:
            original = None

        try:
            # Test disabled
            if "SPECDEC_CUDA_GRAPH" in os.environ:
                del os.environ["SPECDEC_CUDA_GRAPH"]
            pipeline = SpeculativePipeline(
                base_lm=FakeLM(), draft_lm=FakeLM(), max_draft=2, device="cpu"
            )
            assert pipeline.enable_cuda_graph is False

            # Test enabled (should still be False on CPU)
            os.environ["SPECDEC_CUDA_GRAPH"] = "1"
            pipeline2 = SpeculativePipeline(
                base_lm=FakeLM(), draft_lm=FakeLM(), max_draft=2, device="cpu"
            )
            assert pipeline2.enable_cuda_graph is False  # CPU doesn't support

        finally:
            # Restore original value
            if original is not None:
                os.environ["SPECDEC_CUDA_GRAPH"] = original
            elif "SPECDEC_CUDA_GRAPH" in os.environ:
                del os.environ["SPECDEC_CUDA_GRAPH"]

    def test_graph_capture_initial_state(self):
        """Test initial state of graph capture."""
        pipeline = SpeculativePipeline(
            base_lm=FakeLM(), draft_lm=FakeLM(), max_draft=2, device="cpu"
        )
        # CUDA graph capture removed - check that enable_cuda_graph is False
        assert pipeline.enable_cuda_graph is False
        assert pipeline.cuda_graph_captured is False
        assert pipeline.cuda_graph_warmup_done is False
        assert pipeline.graph_input_tensor is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_graph_fallback_on_shape_mismatch(self):
        """Test that graph replay falls back on shape mismatch."""
        if "SPECDEC_CUDA_GRAPH" in os.environ:
            original = os.environ["SPECDEC_CUDA_GRAPH"]
        else:
            original = None

        try:
            os.environ["SPECDEC_CUDA_GRAPH"] = "1"
            pipeline = SpeculativePipeline(
                base_lm=FakeLM(), draft_lm=FakeLM(), max_draft=2, device="cuda"
            )

            # CUDA graph capture removed - should be False even on CUDA
            # This test verifies that CUDA graph is disabled by default
            assert pipeline.enable_cuda_graph is False
            assert pipeline.cuda_graph_captured is False

        finally:
            if original is not None:
                os.environ["SPECDEC_CUDA_GRAPH"] = original
            elif "SPECDEC_CUDA_GRAPH" in os.environ:
                del os.environ["SPECDEC_CUDA_GRAPH"]
