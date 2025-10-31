"""
Structured profiler for Phase 3D GPU optimization metrics.

Provides per-step event timing and structured JSON logging for:
- Draft forward pass time
- Verification forward pass time
- Acceptance check time
- KV cache append time
- Memory usage
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class StructuredProfiler:
    """
    Structured profiler for Phase 3D GPU optimization metrics.

    Tracks per-step timing using CUDA events (when available) or wall-clock time,
    and aggregates results into structured JSON format.
    """

    def __init__(
        self,
        device: str = "auto",
        enable_profiling: bool = False,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize structured profiler.

        Args:
            device: Target device (cuda, mps, cpu, auto)
            enable_profiling: Whether profiling is enabled
            output_dir: Directory for JSON output (default: docs/results/)
        """
        self.device = device
        self.enable_profiling = enable_profiling or (
            os.getenv("SPECDEC_PROFILE", "0").lower() in ("1", "true", "yes")
        )

        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.use_cuda_events = (
            self.device == "cuda"
            and torch.cuda.is_available()
            and self.enable_profiling
        )

        # Setup output directory
        if output_dir is None:
            output_dir = Path("docs/results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.step_metrics: List[Dict[str, Any]] = []
        self.run_metadata: Dict[str, Any] = {}

        # CUDA events (if using CUDA)
        self.draft_start_event: Optional[torch.cuda.Event] = None
        self.draft_end_event: Optional[torch.cuda.Event] = None
        self.verify_start_event: Optional[torch.cuda.Event] = None
        self.verify_end_event: Optional[torch.cuda.Event] = None

        if self.use_cuda_events:
            self.draft_start_event = torch.cuda.Event(enable_timing=True)
            self.draft_end_event = torch.cuda.Event(enable_timing=True)
            self.verify_start_event = torch.cuda.Event(enable_timing=True)
            self.verify_end_event = torch.cuda.Event(enable_timing=True)

    def start_draft_timing(self) -> None:
        """Start timing for draft forward pass."""
        if not self.enable_profiling:
            return

        if self.use_cuda_events and self.draft_start_event is not None:
            self.draft_start_event.record()
        else:
            self._draft_start_wall = time.time()

    def end_draft_timing(self) -> float:
        """
        End timing for draft forward pass.

        Returns:
            Draft forward time in milliseconds
        """
        if not self.enable_profiling:
            return 0.0

        if self.use_cuda_events and self.draft_end_event is not None:
            self.draft_end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = (
                self.draft_start_event.elapsed_time(self.draft_end_event)
                if self.draft_start_event is not None
                else 0.0
            )
        else:
            elapsed_ms = (
                time.time() - getattr(self, "_draft_start_wall", time.time())
            ) * 1000

        return elapsed_ms

    def start_verify_timing(self) -> None:
        """Start timing for verification forward pass."""
        if not self.enable_profiling:
            return

        if self.use_cuda_events and self.verify_start_event is not None:
            self.verify_start_event.record()
        else:
            self._verify_start_wall = time.time()

    def end_verify_timing(self) -> float:
        """
        End timing for verification forward pass.

        Returns:
            Verification forward time in milliseconds
        """
        if not self.enable_profiling:
            return 0.0

        if self.use_cuda_events and self.verify_end_event is not None:
            self.verify_end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = (
                self.verify_start_event.elapsed_time(self.verify_end_event)
                if self.verify_start_event is not None
                else 0.0
            )
        else:
            elapsed_ms = (
                time.time() - getattr(self, "_verify_start_wall", time.time())
            ) * 1000

        return elapsed_ms

    def record_acceptance_time(self, acceptance_time_ms: float) -> None:
        """
        Record acceptance check time.

        Args:
            acceptance_time_ms: Time spent in acceptance check (milliseconds)
        """
        if not self.enable_profiling:
            return
        self._acceptance_time_ms = acceptance_time_ms

    def record_kv_append_time(self, kv_append_time_ms: float) -> None:
        """
        Record KV cache append time.

        Args:
            kv_append_time_ms: Time spent in KV append (milliseconds)
        """
        if not self.enable_profiling:
            return
        self._kv_append_time_ms = kv_append_time_ms

    def record_step(
        self,
        step: int,
        draft_time_ms: float,
        verify_time_ms: float,
        acceptance_time_ms: float = 0.0,
        kv_append_time_ms: float = 0.0,
        accepted_len: int = 0,
        proposed_len: int = 0,
    ) -> None:
        """
        Record metrics for a single step.

        Args:
            step: Step number
            draft_time_ms: Draft forward time (milliseconds)
            verify_time_ms: Verification forward time (milliseconds)
            acceptance_time_ms: Acceptance check time (milliseconds)
            kv_append_time_ms: KV append time (milliseconds)
            accepted_len: Number of accepted tokens
            proposed_len: Number of proposed tokens
        """
        if not self.enable_profiling:
            return

        step_metric = {
            "step": step,
            "draft_forward_time_ms": draft_time_ms,
            "verify_forward_time_ms": verify_time_ms,
            "acceptance_check_time_ms": acceptance_time_ms,
            "kv_append_time_ms": kv_append_time_ms,
            "accepted_len": accepted_len,
            "proposed_len": proposed_len,
        }

        self.step_metrics.append(step_metric)

    def set_run_metadata(
        self,
        device: str,
        dtype: str,
        base_model: str,
        draft_model: str,
        k: int,
        max_tokens: int,
        **kwargs: Any,
    ) -> None:
        """
        Set run metadata.

        Args:
            device: Device name (cuda, mps, cpu)
            dtype: Data type (float16, float32)
            base_model: Base model name
            draft_model: Draft model name
            k: Maximum draft tokens
            max_tokens: Maximum generation tokens
            **kwargs: Additional metadata fields
        """
        self.run_metadata = {
            "device": device,
            "dtype": dtype,
            "base_model": base_model,
            "draft_model": draft_model,
            "k": k,
            "max_tokens": max_tokens,
            **kwargs,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics for current device.

        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        if self.device == "cuda" and torch.cuda.is_available():
            stats["memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
            stats["memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            stats["memory_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (
                1024**2
            )

            # Get detailed stats if available
            if hasattr(torch.cuda, "memory_stats"):
                cuda_stats = torch.cuda.memory_stats()
                stats["memory_peak_mb"] = cuda_stats.get(
                    "allocated_bytes.all.peak", 0
                ) / (1024**2)
        elif self.device == "mps" and hasattr(torch.backends, "mps"):
            # MPS memory tracking
            if hasattr(torch.mps, "driver_allocated_memory"):
                stats["memory_allocated_mb"] = torch.mps.driver_allocated_memory() / (
                    1024**2
                )
            else:
                # Fallback for older PyTorch versions
                stats["memory_allocated_mb"] = 0.0

        return stats

    def aggregate_metrics(self) -> Dict[str, Any]:
        """
        Aggregate step metrics into summary statistics.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.step_metrics:
            return {}

        draft_times = [m["draft_forward_time_ms"] for m in self.step_metrics]
        verify_times = [m["verify_forward_time_ms"] for m in self.step_metrics]
        acceptance_times = [m["acceptance_check_time_ms"] for m in self.step_metrics]
        kv_times = [m["kv_append_time_ms"] for m in self.step_metrics]

        def mean_std(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0}
            import statistics

            return {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        return {
            "draft_forward_time_ms": mean_std(draft_times),
            "verify_forward_time_ms": mean_std(verify_times),
            "acceptance_check_time_ms": mean_std(acceptance_times),
            "kv_append_time_ms": mean_std(kv_times),
            "total_steps": len(self.step_metrics),
            "memory_stats": self.get_memory_stats(),
        }

    def save_json(self, filename: Optional[str] = None) -> Path:
        """
        Save metrics to JSON file.

        Args:
            filename: Output filename (default: auto-generated)

        Returns:
            Path to saved JSON file
        """
        if filename is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phase3d_metrics_{timestamp}.json"

        output_path = self.output_dir / filename

        output_data = {
            "metadata": self.run_metadata,
            "aggregated_metrics": self.aggregate_metrics(),
            "step_metrics": self.step_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Structured metrics saved to {output_path}")
        return output_path

    def reset(self) -> None:
        """Reset profiler state for new run."""
        self.step_metrics = []
        self.run_metadata = {}


def create_structured_profiler(
    device: str = "auto",
    enable_profiling: bool = False,
    output_dir: Optional[Path] = None,
) -> StructuredProfiler:
    """
    Create a structured profiler instance.

    Args:
        device: Target device
        enable_profiling: Whether profiling is enabled
        output_dir: Output directory for JSON files

    Returns:
        StructuredProfiler instance
    """
    return StructuredProfiler(
        device=device, enable_profiling=enable_profiling, output_dir=output_dir
    )
