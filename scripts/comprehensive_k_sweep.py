#!/usr/bin/env python3
"""
Comprehensive K-Sweep Testing Script for Speculative Decoding
Tests K=1-4 with 10 iterations on 10-prompt suite, generates detailed results and plots.
"""

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Silence tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path before importing project modules
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kernels import get_kernel_info  # noqa: E402
from specdec.pipeline import SpeculativePipeline  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 10-prompt test suite
PROMPT_SUITE = [
    "Explain KV cache simply.",
    "What is the capital of France?",
    "Write a short poem about coding.",
    "How does machine learning work?",
    "Describe the process of photosynthesis.",
    "What are the benefits of exercise?",
    "Explain quantum computing basics.",
    "How do neural networks learn?",
    "What is the meaning of life?",
    "Describe a typical day in the life of a programmer.",
]


def get_system_info(device):
    """Get system and environment metadata."""
    # Get kernel backend info
    kinfo = get_kernel_info()

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if device == "cuda" and torch.cuda.is_available()
            else (
                "MPS"
                if device == "mps" and torch.backends.mps.is_available()
                else "CPU"
            )
        ),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "dtype": "float16" if device in ["cuda", "mps"] else "float32",
        "kernel_backends": {
            "verify": kinfo.get("verify_backend", "unknown"),
            "kv_append": kinfo.get("kv_append_backend", "unknown"),
        },
        "kv_append_enabled": os.getenv("SPECDEC_ENABLE_KV_APPEND", "1").lower()
        in ("1", "true", "yes"),
    }


def resolve_device(device_arg):
    """Resolve device argument to actual device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def set_deterministic_mode(enable: bool):
    """Enable reproducible behavior across libraries."""
    if not enable:
        return
    try:
        import random

        import numpy as np
        import torch

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.warning(f"Failed to set deterministic mode: {e}")


def run_comprehensive_k_sweep(
    base_model="gpt2",
    draft_model="distilgpt2",
    max_tokens=32,
    iterations=10,
    device="auto",
    deterministic: bool = False,
):
    """Run comprehensive K-sweep test with 10-prompt suite."""

    # Resolve device
    resolved_device = resolve_device(device)
    logger.info(f"Using device: {resolved_device}")

    # Deterministic mode (optional via flag/env)
    env_det = os.getenv("SPECDEC_DETERMINISTIC", "0").lower() in ("1", "true", "yes")
    deterministic = deterministic or env_det
    set_deterministic_mode(deterministic)
    if deterministic:
        logger.info(
            "Deterministic mode: ON (fixed seeds; cudnn.deterministic; no benchmark)"
        )
    else:
        logger.info("Deterministic mode: OFF")

    # Kernel backend audit (single line)
    kinfo = get_kernel_info()
    dtype_str = "float16" if resolved_device in ["cuda", "mps"] else "float32"
    logger.info(
        "Kernel backends: "
        f"verify={kinfo.get('verify_backend')}, "
        f"kv_append={kinfo.get('kv_append_backend')}, "
        f"device={resolved_device}, dtype={dtype_str}"
    )
    if resolved_device == "cuda" and kinfo.get("verify_backend") != "cuda":
        logger.warning(
            "CUDA requested but verify kernel backend is not CUDA; falling back to "
            f"{kinfo.get('verify_backend')}"
        )

    results = []
    detailed_results = []

    # Cache pipelines per K to avoid reloading models
    pipeline_cache = {}

    for k in range(1, 5):  # K = 1, 2, 3, 4
        logger.info(f"Testing K={k}...")

        # Create pipeline once per K (cached)
        if k not in pipeline_cache:
            logger.info(f"  Creating pipeline for K={k}...")
            try:
                pipeline_cache[k] = SpeculativePipeline(
                    base_model=base_model,
                    draft_model=draft_model,
                    max_draft=k,
                    implementation="hf",
                    enable_optimization=True,
                    enable_profiling=True,
                    device=resolved_device,
                    draft_mode="vanilla",  # disable Medusa/EAGLE paths in sweeps
                )
                logger.info(f"  Pipeline for K={k} created successfully")
            except Exception as e:
                logger.error(f"  Failed to create pipeline for K={k}: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                pipeline_cache[k] = None

        k_results = []
        k_failures = 0

        for iteration in range(iterations):
            logger.info(f"  Iteration {iteration+1}/{iterations}")

            for prompt_idx, prompt in enumerate(PROMPT_SUITE):
                logger.info(f"    Prompt {prompt_idx+1}/10: '{prompt[:30]}...'")

                try:
                    # Use cached pipeline
                    pipeline = pipeline_cache[k]
                    if pipeline is None:
                        raise Exception("Pipeline creation failed for this K")

                    # Generate text
                    start_time = time.time()
                    result = pipeline.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        do_sample=True,
                    )
                    end_time = time.time()

                    # Extract metrics
                    latency_ms = result.get(
                        "latency_ms", (end_time - start_time) * 1000
                    )
                    tokens_per_sec = result.get(
                        "tokens_per_sec", max_tokens / (latency_ms / 1000)
                    )
                    acceptance_rate = result.get("acceptance_rate", 0.0)
                    proposed = result.get("proposed", 0)
                    accepted = result.get("accepted", 0)
                    text = result.get("text", "")
                    kv_appended = result.get("kv_appended_tokens_total", 0)
                    kv_append_time = result.get("kv_append_time_ms", 0.0)
                    kv_append_enabled = result.get("kv_append_enabled", False)
                    kv_append_backend = result.get("kv_append_backend", "unknown")

                    # Store detailed result
                    detailed_result = {
                        "k": k,
                        "iteration": iteration + 1,
                        "prompt_idx": prompt_idx + 1,
                        "prompt": prompt,
                        "latency_ms": latency_ms,
                        "tokens_per_sec": tokens_per_sec,
                        "acceptance_rate": acceptance_rate,
                        "proposed": proposed,
                        "accepted": accepted,
                        "kv_appended_tokens": kv_appended,
                        "kv_append_time_ms": kv_append_time,
                        "kv_append_enabled": kv_append_enabled,
                        "kv_append_backend": kv_append_backend,
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "success": True,
                        "device": resolved_device,
                        "dtype": (
                            "float16"
                            if resolved_device in ["cuda", "mps"]
                            else "float32"
                        ),
                    }
                    detailed_results.append(detailed_result)

                    # Store for K-level aggregation
                    k_results.append(
                        {
                            "latency_ms": latency_ms,
                            "tokens_per_sec": tokens_per_sec,
                            "acceptance_rate": acceptance_rate,
                            "proposed": proposed,
                            "accepted": accepted,
                            "kv_appended_tokens": kv_appended,
                            "kv_append_time_ms": kv_append_time,
                        }
                    )

                    logger.info(
                        f"      K={k}, Iter={iteration+1}, Prompt={prompt_idx+1}: "
                        f"{tokens_per_sec:.2f} tok/s, {acceptance_rate:.3f} accept rate"
                    )

                except Exception as e:
                    k_failures += 1
                    error_traceback = traceback.format_exc()
                    logger.error(
                        f"      K={k}, Iter={iteration+1}, Prompt={prompt_idx+1} "
                        f"failed: {e}"
                    )
                    logger.error(f"      Traceback: {error_traceback}")

                    detailed_result = {
                        "k": k,
                        "iteration": iteration + 1,
                        "prompt_idx": prompt_idx + 1,
                        "prompt": prompt,
                        "error": str(e),
                        "traceback": error_traceback,
                        "success": False,
                        "device": resolved_device,
                        "dtype": (
                            "float16"
                            if resolved_device in ["cuda", "mps"]
                            else "float32"
                        ),
                    }
                    detailed_results.append(detailed_result)

        # Calculate statistics for this K
        valid_results = [r for r in k_results if "error" not in r]
        total_attempts = iterations * len(PROMPT_SUITE)

        if valid_results:
            latencies = [r["latency_ms"] for r in valid_results]
            throughputs = [r["tokens_per_sec"] for r in valid_results]
            acceptance_rates = [r["acceptance_rate"] for r in valid_results]
            proposed_counts = [r["proposed"] for r in valid_results]
            accepted_counts = [r["accepted"] for r in valid_results]
            kv_appended_counts = [r["kv_appended_tokens"] for r in valid_results]
            kv_append_times = [r["kv_append_time_ms"] for r in valid_results]

            results.append(
                {
                    "k": k,
                    "n_samples": len(valid_results),
                    "n_failures": k_failures,
                    "success_rate": len(valid_results) / total_attempts,
                    "latency_ms_mean": np.mean(latencies),
                    "latency_ms_std": np.std(latencies),
                    "tokens_per_sec_mean": np.mean(throughputs),
                    "tokens_per_sec_std": np.std(throughputs),
                    "acceptance_rate_mean": np.mean(acceptance_rates),
                    "acceptance_rate_std": np.std(acceptance_rates),
                    "kv_appended_tokens_mean": np.mean(kv_appended_counts),
                    "kv_appended_tokens_std": np.std(kv_appended_counts),
                    "kv_append_time_ms_mean": np.mean(kv_append_times),
                    "kv_append_time_ms_std": np.std(kv_append_times),
                    "proposed_mean": np.mean(proposed_counts),
                    "proposed_std": np.std(proposed_counts),
                    "accepted_mean": np.mean(accepted_counts),
                    "accepted_std": np.std(accepted_counts),
                    "device": resolved_device,
                    "dtype": (
                        "float16" if resolved_device in ["cuda", "mps"] else "float32"
                    ),
                }
            )

            logger.info(
                f"  K={k} AVERAGE: "
                f"{np.mean(throughputs):.2f}±{np.std(throughputs):.2f} "
                f"tok/s, {np.mean(acceptance_rates):.3f}±"
                f"{np.std(acceptance_rates):.3f} accept rate, "
                f"success rate: {len(valid_results)}/{total_attempts} "
                f"({len(valid_results)/total_attempts:.1%})"
            )
        else:
            # Still write a row even if all failed
            results.append(
                {
                    "k": k,
                    "n_samples": 0,
                    "n_failures": k_failures,
                    "success_rate": 0.0,
                    "latency_ms_mean": float("nan"),
                    "latency_ms_std": float("nan"),
                    "tokens_per_sec_mean": float("nan"),
                    "tokens_per_sec_std": float("nan"),
                    "acceptance_rate_mean": float("nan"),
                    "acceptance_rate_std": float("nan"),
                    "proposed_mean": float("nan"),
                    "proposed_std": float("nan"),
                    "accepted_mean": float("nan"),
                    "accepted_std": float("nan"),
                    "device": resolved_device,
                    "dtype": (
                        "float16" if resolved_device in ["cuda", "mps"] else "float32"
                    ),
                }
            )
            logger.error(
                f"  K={k}: All {total_attempts} attempts failed ({k_failures} failures)"
            )

    return (
        results,
        detailed_results,
        {"kernel_info": kinfo, "deterministic": deterministic},
    )


def save_results(results, detailed_results, system_info, output_dir, device):
    """Save results to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get detailed metrics if available
    detailed_metrics = {}
    try:
        from src.metrics.detailed_profiler import get_summary

        detailed_metrics = get_summary()
    except ImportError:
        pass

    # Add detailed metrics to system info
    if detailed_metrics and detailed_metrics.get("enabled", False):
        system_info["detailed_metrics"] = detailed_metrics

    # Save summary results to CSV
    csv_file = output_dir / f"specdec_{device}_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save detailed results to JSON
    json_file = output_dir / f"specdec_{device}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "system_info": system_info,
                "summary_results": results,
                "detailed_results": detailed_results,
                "detailed_metrics": detailed_metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {csv_file} and {json_file}")
    return csv_file, json_file


def create_plots(results, output_dir):
    """Create performance plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Create figures directory
        figures_dir = Path(output_dir).parent / "docs" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        k_values = [r["k"] for r in results]
        throughput_means = [r["tokens_per_sec_mean"] for r in results]
        throughput_stds = [r["tokens_per_sec_std"] for r in results]
        acceptance_means = [r["acceptance_rate_mean"] for r in results]
        acceptance_stds = [r["acceptance_rate_std"] for r in results]

        # Plot 1: Tokens/sec vs K
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            k_values,
            throughput_means,
            yerr=throughput_stds,
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("K (Draft Tokens)", fontsize=12)
        plt.ylabel("Throughput (tokens/sec)", fontsize=12)
        plt.title("Speculative Decoding Performance: Throughput vs K", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)

        # Add value labels
        for i, (k, mean, std) in enumerate(
            zip(k_values, throughput_means, throughput_stds)
        ):
            plt.annotate(
                f"{mean:.2f}±{std:.2f}",
                (k, mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "throughput_vs_k.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 2: Acceptance Rate vs K
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            k_values,
            acceptance_means,
            yerr=acceptance_stds,
            marker="s",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
            color="orange",
        )
        plt.xlabel("K (Draft Tokens)", fontsize=12)
        plt.ylabel("Acceptance Rate", fontsize=12)
        plt.title("Speculative Decoding Performance: Acceptance Rate vs K", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)

        # Add value labels
        for i, (k, mean, std) in enumerate(
            zip(k_values, acceptance_means, acceptance_stds)
        ):
            plt.annotate(
                f"{mean:.3f}±{std:.3f}",
                (k, mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "acceptance_vs_k.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Plots saved to {figures_dir}")

    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
    except Exception as e:
        logger.error(f"Error creating plots: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive K-Sweep Testing for Speculative Decoding"
    )
    parser.add_argument("--base-model", default="gpt2", help="Base model name")
    parser.add_argument("--draft-model", default="distilgpt2", help="Draft model name")
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

    args = parser.parse_args()

    # Set environment variables for detailed metrics
    if args.metrics_detailed:
        os.environ["SPECDEC_DETAILED_METRICS"] = "1"

    if args.cuda_graph:
        os.environ["SPECDEC_CUDA_GRAPH"] = "1"

    # Resolve device
    resolved_device = resolve_device(args.device)

    logger.info(
        f"Starting comprehensive K-sweep test: {args.base_model} + {args.draft_model}"
    )
    logger.info(f"Device: {resolved_device} (requested: {args.device})")
    logger.info(f"Max tokens: {args.max_tokens}, Iterations: {args.iterations}")
    logger.info(f"Prompt suite: {len(PROMPT_SUITE)} prompts")

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

    # Print summary table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE K-SWEEP RESULTS SUMMARY")
    print("=" * 120)
    print(
        f"Device: {resolved_device} | Dtype: {system_info['dtype']} | "
        f"Models: {args.base_model} + {args.draft_model}"
    )
    print("=" * 120)
    print(
        f"{'K':<3} {'Samples':<8} {'Failures':<9} {'Success%':<9} "
        f"{'Latency (ms)':<20} {'Throughput (tok/s)':<20} "
        f"{'Accept Rate':<15} {'Proposed':<12} {'Accepted':<12}"
    )
    print("-" * 120)

    for result in results:
        if result["n_samples"] > 0:
            print(
                f"{result['k']:<3} {result['n_samples']:<8} {result['n_failures']:<9} "
                f"{result['success_rate']*100:.1f}%{'':<4} "
                f"{result['latency_ms_mean']:.1f}±{result['latency_ms_std']:.1f}    "
                f"{result['tokens_per_sec_mean']:.2f}±"
                f"{result['tokens_per_sec_std']:.2f}        "
                f"{result['acceptance_rate_mean']:.3f}±"
                f"{result['acceptance_rate_std']:.3f}    "
                f"{result['proposed_mean']:.1f}±{result['proposed_std']:.1f}    "
                f"{result['accepted_mean']:.1f}±{result['accepted_std']:.1f}"
            )
        else:
            print(
                f"{result['k']:<3} {result['n_samples']:<8} {result['n_failures']:<9} "
                f"{result['success_rate']*100:.1f}%{'':<4} "
                f"{'N/A':<20} {'N/A':<20} {'N/A':<15} {'N/A':<12} {'N/A':<12}"
            )

    print("=" * 120)
    logger.info(
        f"Comprehensive K-sweep completed. Results saved to {csv_file} and {json_file}"
    )


if __name__ == "__main__":
    main()
