#!/usr/bin/env python3
"""
Comprehensive K-Sweep Testing Script for Speculative Decoding

This script uses the unified k_sweep module for consistent result formatting.
All k_sweep scripts now produce results in the same format for easy comparison.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Force unbuffered output for Kaggle/notebook environments
(
    sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stdout, "reconfigure")
    else None
)
(
    sys.stderr.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure")
    else None
)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path before importing project modules
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Configure PyTorch for CUDA inference
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=True)

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        device_capability = torch.cuda.get_device_capability(0)
        compute_capability = f"{device_capability[0]}{device_capability[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = compute_capability

    if os.getenv("CUDA_LAUNCH_BLOCKING") == "1":
        print(
            "[STARTUP] WARNING: CUDA_LAUNCH_BLOCKING=1 is set - "
            "async streams will be disabled! "
            "This will reduce GPU utilization and throughput. "
            "Remove for production runs.",
            flush=True,
        )
    else:
        print(
            "[STARTUP] CUDA_LAUNCH_BLOCKING not set - "
            "async streams enabled for optimal performance",
            flush=True,
        )

    try:
        import torch._dynamo  # noqa: E402

        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
    except ImportError:
        pass

import logging  # noqa: E402

from k_sweep import (  # noqa: E402
    PROMPT_SUITE,
    create_plots,
    get_system_info,
    resolve_device,
    run_comprehensive_k_sweep,
    save_results,
)

# Import from unified k_sweep module
from kernels import get_kernel_info  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Ensure initial output is visible
print("=" * 80, flush=True)
print("[SCRIPT] Comprehensive K-Sweep Script Starting", flush=True)
print("=" * 80, flush=True)
sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive K-Sweep Testing for Speculative Decoding"
    )
    parser.add_argument("--base-model", default="gpt2", help="Base model name")
    parser.add_argument("--draft-model", default="distilgpt2", help="Draft model name")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (default: only show results)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations per K"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to run on (auto selects best available)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--metrics-detailed",
        action="store_true",
        help="Enable detailed metrics collection",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture (CUDA only)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic mode "
            "(seeds, cudnn.deterministic, disable experimental draftors)"
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=4,
        help="Maximum K value to sweep (1..max_k)",
    )
    parser.add_argument(
        "--single-prompt",
        type=str,
        default=None,
        help="Use single prompt instead of prompt suite (for simple/T4 mode)",
    )
    parser.add_argument(
        "--t4-warmup",
        action="store_true",
        help="Enable T4-specific warmup and memory management",
    )
    parser.add_argument(
        "--reuse-pipeline",
        action="store_true",
        help="Reuse pipeline across K values (more efficient, T4-style)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: from SPECDEC_BATCH_SIZE or 8)",
    )

    args = parser.parse_args()

    # Set environment variables for detailed metrics
    if args.metrics_detailed:
        os.environ["SPECDEC_DETAILED_METRICS"] = "1"

    if args.cuda_graph:
        os.environ["SPECDEC_CUDA_GRAPH"] = "1"

    # Resolve device
    resolved_device = resolve_device(args.device)

    # Check if baseline mode (no draft model)
    is_baseline = args.draft_model.lower() in ("none", "")
    if is_baseline:
        logger.info(
            f"Running non-speculative baseline (no draft model): {args.base_model}"
        )
    else:
        logger.info(
            f"Starting comprehensive K-sweep test: {args.base_model} + {args.draft_model}"
        )
    logger.info(f"Device: {resolved_device} (requested: {args.device})")
    logger.info(
        f"Max tokens: {args.max_tokens}, Iterations: {args.iterations}, Max K: {args.max_k}"
    )

    # Determine prompt mode
    if args.single_prompt:
        logger.info(f"Mode: Single prompt - '{args.single_prompt}'")
        prompt_mode = "single"
    else:
        logger.info(f"Mode: Prompt suite - {len(PROMPT_SUITE)} prompts")
        prompt_mode = "suite"

    if args.t4_warmup:
        logger.info("T4 optimizations: Enabled (warmup, memory management)")
    if args.reuse_pipeline:
        logger.info("Pipeline reuse: Enabled (more efficient across K values)")
    if args.batch_size:
        os.environ["SPECDEC_BATCH_SIZE"] = str(args.batch_size)
        logger.info(f"Batch size: {args.batch_size} (override)")

    # Get system info (add kernel info + deterministic)
    system_info = get_system_info(resolved_device)
    system_info["deterministic"] = args.deterministic or (
        os.getenv("SPECDEC_DETERMINISTIC", "0").lower() in ("1", "true", "yes")
    )
    system_info["kernel_backends"] = get_kernel_info()

    # Run tests
    results, detailed_results, run_meta = run_comprehensive_k_sweep(
        base_model=args.base_model,
        draft_model=args.draft_model,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
        device=args.device,
        deterministic=args.deterministic,
        verbose=args.verbose,
        max_k=args.max_k,
        single_prompt=args.single_prompt,
        use_prompt_suite=(prompt_mode == "suite"),
        t4_warmup=args.t4_warmup,
        reuse_pipeline=args.reuse_pipeline,
    )

    # Save results
    # Merge run metadata into system info for JSON
    system_info.update(run_meta)

    csv_file, json_file = save_results(
        results, detailed_results, system_info, args.output_dir, resolved_device
    )

    # Create plots (unless disabled)
    if not args.no_plots:
        create_plots(results, args.output_dir)
    else:
        logger.info("Skipping plot generation (--no-plots specified)")

    # Print summary table with real values
    print("\n" + "=" * 120, flush=True)
    print("K-SWEEP RESULTS SUMMARY", flush=True)
    print("=" * 120, flush=True)
    print(
        f"Device: {resolved_device} | Dtype: {system_info['dtype']} | "
        f"Models: {args.base_model} + {args.draft_model}",
        flush=True,
    )
    print("=" * 120, flush=True)
    print(
        f"{'K':<3} {'Samples':<8} {'Failures':<9} {'Success%':<9} "
        f"{'Latency (ms)':<20} {'Throughput (tok/s)':<20} "
        f"{'Accept Rate':<15} {'Proposed':<12} {'Accepted':<12}",
        flush=True,
    )
    print("-" * 120, flush=True)

    for result in results:
        if result["n_samples"] > 0:
            # Validate values before printing
            latency_mean = result.get("latency_ms_mean", 0.0)
            latency_std = result.get("latency_ms_std", 0.0)
            throughput_mean = result.get("tokens_per_sec_mean", 0.0)
            throughput_std = result.get("tokens_per_sec_std", 0.0)
            accept_mean = result.get("acceptance_rate_mean", 0.0)
            accept_std = result.get("acceptance_rate_std", 0.0)
            proposed_mean = result.get("proposed_mean", 0.0)
            proposed_std = result.get("proposed_std", 0.0)
            accepted_mean = result.get("accepted_mean", 0.0)
            accepted_std = result.get("accepted_std", 0.0)

            # Check for NaN/inf
            if np.isnan(latency_mean) or np.isinf(latency_mean):
                latency_mean = 0.0
            if np.isnan(latency_std) or np.isinf(latency_std):
                latency_std = 0.0
            if np.isnan(throughput_mean) or np.isinf(throughput_mean):
                throughput_mean = 0.0
            if np.isnan(throughput_std) or np.isinf(throughput_std):
                throughput_std = 0.0

            print(
                f"{result['k']:<3} {result['n_samples']:<8} {result['n_failures']:<9} "
                f"{result['success_rate']*100:.1f}%{'':<4} "
                f"{latency_mean:.1f}±{latency_std:.1f}{'':<15} "
                f"{throughput_mean:.2f}±{throughput_std:.2f}{'':<12} "
                f"{accept_mean:.3f}±{accept_std:.3f}{'':<9} "
                f"{proposed_mean:.1f}±{proposed_std:.1f}{'':<7} "
                f"{accepted_mean:.1f}±{accepted_std:.1f}",
                flush=True,
            )
        else:
            print(
                f"{result['k']:<3} {result['n_samples']:<8} {result['n_failures']:<9} "
                f"{result['success_rate']*100:.1f}%{'':<4} "
                f"{'N/A':<20} {'N/A':<20} {'N/A':<15} {'N/A':<12} {'N/A':<12}",
                flush=True,
            )

    print("=" * 120, flush=True)
    print(
        f"\n[FINAL] Comprehensive K-sweep completed. Results saved to {csv_file} and {json_file}",
        flush=True,
    )
    logger.info(
        f"Comprehensive K-sweep completed. Results saved to {csv_file} and {json_file}"
    )
    sys.stdout.flush()
    sys.stderr.flush()

    # Cleanup CUDA resources to ensure clean exit in Kaggle
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("[CLEANUP] CUDA resources cleaned up", flush=True)

    # Explicit exit to ensure script completes in Kaggle notebooks
    print("[SCRIPT] Script completed successfully", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SCRIPT] Interrupted by user", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}", flush=True)
        logger.exception("Fatal error in main()")
        sys.exit(1)
