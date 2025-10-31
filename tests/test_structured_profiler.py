"""
Tests for structured profiler and metrics JSON keys.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.metrics.structured_profiler import StructuredProfiler


class TestStructuredProfiler:
    """Test structured profiler functionality."""

    def test_profiler_initialization(self):
        """Test profiler initializes correctly."""
        profiler = StructuredProfiler(device="cpu", enable_profiling=False)
        assert profiler.device == "cpu"
        assert profiler.enable_profiling is False
        assert len(profiler.step_metrics) == 0

    def test_profiler_record_step(self):
        """Test recording step metrics."""
        profiler = StructuredProfiler(device="cpu", enable_profiling=True)
        profiler.record_step(
            step=1,
            draft_time_ms=5.0,
            verify_time_ms=8.0,
            acceptance_time_ms=0.5,
            kv_append_time_ms=1.0,
            accepted_len=2,
            proposed_len=4,
        )

        assert len(profiler.step_metrics) == 1
        assert profiler.step_metrics[0]["step"] == 1
        assert profiler.step_metrics[0]["draft_forward_time_ms"] == 5.0
        assert profiler.step_metrics[0]["verify_forward_time_ms"] == 8.0
        assert profiler.step_metrics[0]["kv_append_time_ms"] == 1.0

    def test_profiler_aggregate_metrics(self):
        """Test metric aggregation."""
        profiler = StructuredProfiler(device="cpu", enable_profiling=True)
        profiler.record_step(step=1, draft_time_ms=5.0, verify_time_ms=8.0)
        profiler.record_step(step=2, draft_time_ms=6.0, verify_time_ms=9.0)

        aggregated = profiler.aggregate_metrics()
        assert aggregated["total_steps"] == 2
        assert "draft_forward_time_ms" in aggregated
        assert aggregated["draft_forward_time_ms"]["mean"] == 5.5
        assert aggregated["verify_forward_time_ms"]["mean"] == 8.5

    def test_profiler_save_json(self):
        """Test saving metrics to JSON with required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = StructuredProfiler(
                device="cpu", enable_profiling=True, output_dir=tmpdir
            )
            profiler.set_run_metadata(
                device="cpu",
                dtype="float32",
                base_model="gpt2",
                draft_model="distilgpt2",
                k=4,
                max_tokens=32,
            )
            profiler.record_step(step=1, draft_time_ms=5.0, verify_time_ms=8.0)

            output_path = profiler.save_json("test_metrics.json")

            assert output_path.exists()

            # Load and verify JSON structure
            with open(output_path) as f:
                data = json.load(f)

            # Verify required keys exist
            assert "metadata" in data
            assert "aggregated_metrics" in data
            assert "step_metrics" in data

            # Verify metadata keys
            assert "device" in data["metadata"]
            assert "dtype" in data["metadata"]
            assert "base_model" in data["metadata"]
            assert "draft_model" in data["metadata"]
            assert "k" in data["metadata"]
            assert "max_tokens" in data["metadata"]

            # Verify aggregated metrics structure
            assert "draft_forward_time_ms" in data["aggregated_metrics"]
            assert "verify_forward_time_ms" in data["aggregated_metrics"]
            assert "total_steps" in data["aggregated_metrics"]

    def test_profiler_memory_stats_cpu(self):
        """Test memory stats on CPU."""
        profiler = StructuredProfiler(device="cpu", enable_profiling=True)
        stats = profiler.get_memory_stats()
        assert isinstance(stats, dict)

    def test_profiler_disabled_no_recording(self):
        """Test that disabled profiler doesn't record."""
        profiler = StructuredProfiler(device="cpu", enable_profiling=False)
        profiler.record_step(step=1, draft_time_ms=5.0, verify_time_ms=8.0)
        assert len(profiler.step_metrics) == 0
