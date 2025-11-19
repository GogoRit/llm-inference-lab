#!/usr/bin/env python3
"""
T4-Optimized K-Sweep Benchmark for Zero-Copy Speculative Decoding

Profiles performance on Tesla T4 (16GB VRAM) with:
- Pre-allocated KV cache buffers
- Zero-copy pointer rollback
- Proper warmup and synchronization
- TPS, acceptance rate, and overhead metrics
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Force unbuffered output
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

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import torch

# Configure PyTorch for CUDA inference
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from specdec import SpeculativePipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clear_gpu_cache():
    """Clear GPU cache to prevent OOM on T4."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def warmup_run(
    pipeline: SpeculativePipeline,
    prompt: str,
    warmup_tokens: int = 5,
) -> None:
    """
    Perform warmup run to compile/optimize GPU kernels.

    Args:
        pipeline: Initialized pipeline
        prompt: Test prompt
        warmup_tokens: Number of tokens to generate for warmup
    """
    logger.info(f"Warming up GPU kernels with {warmup_tokens} tokens...")

    try:
        # Enable KV cache for warmup
        os.environ["SPECDEC_ENABLE_KV_APPEND"] = "1"

        # Run warmup generation with greedy decoding (deterministic)
        _ = pipeline.generate_batch(
            prompts=[prompt],
            max_tokens=warmup_tokens,
            temperature=1e-5,  # Near-zero for greedy (avoid divide-by-zero)
            do_sample=False,  # Greedy decoding
            top_p=1.0,  # Disable top-p sampling
        )

        # Synchronize to ensure warmup completes
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("Warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")
    finally:
        # Clear cache after warmup
        clear_gpu_cache()


def run_benchmark(
    pipeline: SpeculativePipeline,
    prompt: str,
    k: int,
    max_tokens: int,
    iterations: int = 3,
) -> Dict[str, float]:
    """
    Run benchmark for a specific K value.

    Args:
        pipeline: Initialized pipeline
        prompt: Test prompt
        k: Draft token count (K)
        max_tokens: Maximum tokens to generate
        iterations: Number of iterations to average

    Returns:
        Dictionary with metrics: tps, acceptance_rate, draft_time_ms, verify_time_ms
    """
    logger.info(f"Benchmarking K={k} ({iterations} iterations)...")

    # Set K via controller
    # The controller should be configured to use fixed K for benchmarking
    # We'll pass k via controller_params if needed, but the controller
    # should already be set up with fixed K mode
    if hasattr(pipeline, "controller") and pipeline.controller is not None:
        # Try to set fixed K for this benchmark
        if hasattr(pipeline.controller, "set_fixed_k"):
            pipeline.controller.set_fixed_k(k)
        elif hasattr(pipeline.controller, "k"):
            pipeline.controller.k = k
        elif hasattr(pipeline.controller, "max_draft"):
            pipeline.controller.max_draft = k
        # If controller is adaptive, we'll need to pass k via context
        # For now, assume fixed controller is used

    all_tps = []
    all_acceptance_rates = []
    all_draft_times = []
    all_verify_times = []
    all_proposed = []
    all_accepted = []

    for iteration in range(iterations):
        logger.info(f"  Iteration {iteration + 1}/{iterations}")

        # Clear cache before each iteration
        clear_gpu_cache()

        try:
            # Force synchronization before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Time the generation
            start_time = time.time()

            # Enforce greedy decoding for deterministic acceptance rates
            # This ensures draft and base models use the same sampling strategy
            results = pipeline.generate_batch(
                prompts=[prompt],
                max_tokens=max_tokens,
                temperature=1e-5,  # Near-zero for greedy (avoid divide-by-zero)
                do_sample=False,  # Greedy decoding
                top_p=1.0,  # Disable top-p sampling
            )

            # Force synchronization after generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            # Extract metrics from result
            result = results[0] if results else {}

            # Calculate TPS (use result's TPS if available, otherwise calculate)
            elapsed_time = end_time - start_time
            tps = result.get(
                "tokens_per_sec", result.get("throughput_tokens_per_sec", 0.0)
            )
            if tps == 0.0:
                # Fallback: calculate from generated tokens
                generated_tokens = len(result.get("generated_tokens", []))
                tps = generated_tokens / elapsed_time if elapsed_time > 0 else 0.0
            all_tps.append(tps)

            # Extract acceptance metrics
            acceptance_rate = result.get("acceptance_rate", 0.0)
            proposed = result.get("proposed", 0)
            accepted = result.get("accepted", 0)

            # If not in result, try batch_metrics
            if proposed == 0 or accepted == 0:
                batch_metrics = result.get("batch_metrics", {})
                proposed = batch_metrics.get("total_proposed", proposed)
                accepted = batch_metrics.get("total_accepted", accepted)
                if acceptance_rate == 0.0 and proposed > 0:
                    acceptance_rate = accepted / proposed

            all_acceptance_rates.append(acceptance_rate)
            all_proposed.append(proposed)
            all_accepted.append(accepted)

            # Extract timing breakdown
            # Try per-prompt averages first, then batch totals
            draft_time = result.get("draft_avg_ms", 0.0)
            verify_time = result.get("verify_avg_ms", 0.0)

            # If not available, try batch_metrics
            if draft_time == 0.0 or verify_time == 0.0:
                batch_metrics = result.get("batch_metrics", {})
                total_steps = batch_metrics.get("total_steps", 1)
                if draft_time == 0.0:
                    draft_time = batch_metrics.get("total_draft_time_ms", 0.0) / max(
                        total_steps, 1
                    )
                if verify_time == 0.0:
                    verify_time = batch_metrics.get(
                        "total_verification_time_ms", 0.0
                    ) / max(total_steps, 1)

            all_draft_times.append(draft_time)
            all_verify_times.append(verify_time)

            logger.info(
                f"    K={k}, Iter={iteration + 1}: "
                f"TPS={tps:.2f}, AcceptRate={acceptance_rate:.3f}, "
                f"Draft={draft_time:.1f}ms, Verify={verify_time:.1f}ms"
            )

        except Exception as e:
            logger.error(f"    K={k}, Iter={iteration + 1} failed: {e}")
            import traceback

            traceback.print_exc()
            # Continue with other iterations

    # Calculate averages
    if not all_tps:
        logger.error(f"K={k}: All iterations failed")
        return {
            "k": k,
            "tps": 0.0,
            "acceptance_rate": 0.0,
            "draft_time_ms": 0.0,
            "verify_time_ms": 0.0,
            "proposed": 0.0,
            "accepted": 0.0,
            "iterations": 0,
        }

    return {
        "k": k,
        "tps": sum(all_tps) / len(all_tps),
        "acceptance_rate": sum(all_acceptance_rates) / len(all_acceptance_rates),
        "draft_time_ms": sum(all_draft_times) / len(all_draft_times),
        "verify_time_ms": sum(all_verify_times) / len(all_verify_times),
        "proposed": sum(all_proposed) / len(all_proposed),
        "accepted": sum(all_accepted) / len(all_accepted),
        "iterations": len(all_tps),
    }


def run_k_sweep(
    base_model: str = "gpt2",
    draft_model: str = "distilgpt2",
    prompt: str = "The future of AI is",
    max_tokens: int = 50,
    k_range: tuple = (1, 9),  # K from 1 to 8
    iterations: int = 3,
    warmup_tokens: int = 5,
    enable_kv_cache: bool = True,
    device: str = "cuda",
) -> List[Dict[str, float]]:
    """
    Run K-sweep benchmark.

    Args:
        base_model: Base model name
        draft_model: Draft model name
        prompt: Test prompt
        max_tokens: Maximum tokens to generate
        k_range: Tuple of (min_k, max_k) - K values to test
        iterations: Number of iterations per K
        warmup_tokens: Number of tokens for warmup
        enable_kv_cache: Whether to enable KV cache
        device: Device to use

    Returns:
        List of benchmark results for each K
    """
    logger.info("=" * 80)
    logger.info("T4 K-SWEEP BENCHMARK - Zero-Copy Speculative Decoding")
    logger.info("=" * 80)
    logger.info(f"Base Model: {base_model}")
    logger.info(f"Draft Model: {draft_model}")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Max Tokens: {max_tokens}")
    logger.info(f"K Range: {k_range[0]} to {k_range[1] - 1}")
    logger.info(f"Iterations per K: {iterations}")
    logger.info(f"KV Cache: {'Enabled' if enable_kv_cache else 'Disabled'}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # Set environment variable for KV cache
    if enable_kv_cache:
        os.environ["SPECDEC_ENABLE_KV_APPEND"] = "1"
    else:
        os.environ.pop("SPECDEC_ENABLE_KV_APPEND", None)

    # Initialize pipeline once (outside timing loop)
    logger.info("Initializing pipeline...")
    try:
        # Use fixed controller for consistent K values during benchmark
        pipeline = SpeculativePipeline(
            base_model=base_model,
            draft_model=draft_model,
            max_draft=k_range[1] - 1,  # Set to max K we'll test
            device=device,
            implementation="hf",
            enable_optimization=True,
            enable_profiling=False,  # Disable profiling for benchmark
            controller="fixed",  # Use fixed controller for benchmarking
            controller_params={"k": k_range[1] - 1},  # Will be overridden per K
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback

        traceback.print_exc()
        return []

    # Warmup run
    warmup_run(pipeline, prompt, warmup_tokens)

    # Run K-sweep
    results = []
    k_min, k_max = k_range

    for k in range(k_min, k_max):
        logger.info("")
        logger.info(f"Testing K={k}...")

        # Run benchmark for this K
        result = run_benchmark(
            pipeline=pipeline,
            prompt=prompt,
            k=k,
            max_tokens=max_tokens,
            iterations=iterations,
        )

        results.append(result)

        # Clear cache between K values
        clear_gpu_cache()

    # Calculate speedup vs K=1
    k1_tps = next((r["tps"] for r in results if r["k"] == 1), None)
    if k1_tps and k1_tps > 0:
        for result in results:
            result["speedup_vs_k1"] = result["tps"] / k1_tps
    else:
        for result in results:
            result["speedup_vs_k1"] = 0.0

    return results


def print_results_table(results: List[Dict[str, float]]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("K-SWEEP RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'K':<4} {'TPS':<10} {'AcceptRate':<12} {'Speedup':<10} "
        f"{'Draft(ms)':<12} {'Verify(ms)':<12} {'Proposed':<10} {'Accepted':<10}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result['k']:<4} "
            f"{result['tps']:<10.2f} "
            f"{result['acceptance_rate']:<12.3f} "
            f"{result.get('speedup_vs_k1', 0.0):<10.2f} "
            f"{result['draft_time_ms']:<12.1f} "
            f"{result['verify_time_ms']:<12.1f} "
            f"{result['proposed']:<10.1f} "
            f"{result['accepted']:<10.1f}"
        )

    print("=" * 100)


def save_csv(results: List[Dict[str, float]], output_file: str) -> None:
    """Save results to CSV file."""
    if not results:
        logger.warning("No results to save")
        return

    fieldnames = [
        "k",
        "tps",
        "acceptance_rate",
        "speedup_vs_k1",
        "draft_time_ms",
        "verify_time_ms",
        "proposed",
        "accepted",
        "iterations",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="T4-Optimized K-Sweep Benchmark for Zero-Copy Speculative Decoding"
    )
    parser.add_argument(
        "--base-model",
        default="gpt2",
        help="Base model name (default: gpt2)",
    )
    parser.add_argument(
        "--draft-model",
        default="distilgpt2",
        help="Draft model name (default: distilgpt2)",
    )
    parser.add_argument(
        "--prompt",
        default="The future of AI is",
        help="Test prompt (default: 'The future of AI is')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=1,
        help="Minimum K value (default: 1)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=9,
        help="Maximum K value (exclusive, default: 9, so K=1-8)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per K (default: 3)",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=5,
        help="Number of tokens for warmup (default: 5)",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Disable KV cache (default: enabled)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--output",
        default="t4_k_sweep_results.csv",
        help="Output CSV file (default: t4_k_sweep_results.csv)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA not available! Use --device cpu or install CUDA.")
        sys.exit(1)

    # Print GPU info if CUDA
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Run K-sweep
    results = run_k_sweep(
        base_model=args.base_model,
        draft_model=args.draft_model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        k_range=(args.k_min, args.k_max),
        iterations=args.iterations,
        warmup_tokens=args.warmup_tokens,
        enable_kv_cache=not args.disable_kv_cache,
        device=args.device,
    )

    if not results:
        logger.error("No results collected!")
        sys.exit(1)

    # Print results table
    print_results_table(results)

    # Save to CSV
    save_csv(results, args.output)

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
