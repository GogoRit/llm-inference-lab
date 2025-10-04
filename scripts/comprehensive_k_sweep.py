#!/usr/bin/env python3
"""
Comprehensive K-Sweep Testing Script for Speculative Decoding
Tests K=1-4 with 10 iterations on 10-prompt suite, generates detailed results and plots.
"""

import argparse
import csv
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec.pipeline import SpeculativePipeline

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


def get_system_info():
    """Get system and environment metadata."""
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "MPS" if torch.backends.mps.is_available() else "CPU"
        ),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def run_comprehensive_k_sweep(
    base_model="gpt2", draft_model="distilgpt2", max_tokens=32, iterations=10
):
    """Run comprehensive K-sweep test with 10-prompt suite."""

    results = []
    detailed_results = []

    for k in range(1, 5):  # K = 1, 2, 3, 4
        logger.info(f"Testing K={k}...")

        k_results = []

        for iteration in range(iterations):
            logger.info(f"  Iteration {iteration+1}/{iterations}")

            for prompt_idx, prompt in enumerate(PROMPT_SUITE):
                logger.info(f"    Prompt {prompt_idx+1}/10: '{prompt[:30]}...'")

                try:
                    # Initialize pipeline with specific K
                    pipeline = SpeculativePipeline(
                        base_model=base_model,
                        draft_model=draft_model,
                        max_draft=k,
                        implementation="hf",
                        enable_optimization=True,
                        enable_profiling=True,
                        device="mps",
                    )

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
                    latency_ms = result.get("latency_ms", (end_time - start_time) * 1000)
                    tokens_per_sec = result.get("tokens_per_sec", max_tokens / (latency_ms / 1000))
                    acceptance_rate = result.get("acceptance_rate", 0.0)
                    proposed = result.get("proposed", 0)
                    accepted = result.get("accepted", 0)
                    text = result.get("text", "")

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
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "success": True,
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
                        }
                    )

                    logger.info(
                        f"      K={k}, Iter={iteration+1}, Prompt={prompt_idx+1}: "
                        f"{tokens_per_sec:.2f} tok/s, {acceptance_rate:.3f} accept rate"
                    )

                except Exception as e:
                    logger.error(
                        f"      K={k}, Iter={iteration+1}, Prompt={prompt_idx+1} failed: {e}"
                    )
                    detailed_result = {
                        "k": k,
                        "iteration": iteration + 1,
                        "prompt_idx": prompt_idx + 1,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False,
                    }
                    detailed_results.append(detailed_result)

        # Calculate statistics for this K
        valid_results = [r for r in k_results if "error" not in r]
        if valid_results:
            latencies = [r["latency_ms"] for r in valid_results]
            throughputs = [r["tokens_per_sec"] for r in valid_results]
            acceptance_rates = [r["acceptance_rate"] for r in valid_results]
            proposed_counts = [r["proposed"] for r in valid_results]
            accepted_counts = [r["accepted"] for r in valid_results]

            results.append(
                {
                    "k": k,
                    "n_samples": len(valid_results),
                    "latency_ms_mean": np.mean(latencies),
                    "latency_ms_std": np.std(latencies),
                    "tokens_per_sec_mean": np.mean(throughputs),
                    "tokens_per_sec_std": np.std(throughputs),
                    "acceptance_rate_mean": np.mean(acceptance_rates),
                    "acceptance_rate_std": np.std(acceptance_rates),
                    "proposed_mean": np.mean(proposed_counts),
                    "proposed_std": np.std(proposed_counts),
                    "accepted_mean": np.mean(accepted_counts),
                    "accepted_std": np.std(accepted_counts),
                }
            )

            logger.info(
                f"  K={k} AVERAGE: {np.mean(throughputs):.2f}±{np.std(throughputs):.2f} "
                f"tok/s, {np.mean(acceptance_rates):.3f}±{np.std(acceptance_rates):.3f} "
                f"accept rate"
            )
        else:
            logger.error(f"  K={k}: All iterations failed")

    return results, detailed_results


def save_results(results, detailed_results, system_info, output_dir):
    """Save results to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary results to CSV
    csv_file = output_dir / f"specdec_cpu_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save detailed results to JSON
    json_file = output_dir / f"specdec_cpu_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "system_info": system_info,
                "summary_results": results,
                "detailed_results": detailed_results,
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
        for i, (k, mean, std) in enumerate(zip(k_values, throughput_means, throughput_stds)):
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
        for i, (k, mean, std) in enumerate(zip(k_values, acceptance_means, acceptance_stds)):
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
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per K")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")

    args = parser.parse_args()

    logger.info(f"Starting comprehensive K-sweep test: {args.base_model} + {args.draft_model}")
    logger.info(f"Max tokens: {args.max_tokens}, Iterations: {args.iterations}")
    logger.info(f"Prompt suite: {len(PROMPT_SUITE)} prompts")

    # Get system info
    system_info = get_system_info()

    # Run tests
    results, detailed_results = run_comprehensive_k_sweep(
        base_model=args.base_model,
        draft_model=args.draft_model,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
    )

    # Save results
    csv_file, json_file = save_results(results, detailed_results, system_info, args.output_dir)

    # Create plots
    create_plots(results, args.output_dir)

    # Print summary table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE K-SWEEP RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'K':<3} {'Samples':<8} {'Latency (ms)':<20} {'Throughput (tok/s)':<20} "
        f"{'Accept Rate':<15} {'Proposed':<12} {'Accepted':<12}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result['k']:<3} {result['n_samples']:<8} "
            f"{result['latency_ms_mean']:.1f}±{result['latency_ms_std']:.1f}    "
            f"{result['tokens_per_sec_mean']:.2f}±{result['tokens_per_sec_std']:.2f}        "
            f"{result['acceptance_rate_mean']:.3f}±{result['acceptance_rate_std']:.3f}    "
            f"{result['proposed_mean']:.1f}±{result['proposed_std']:.1f}    "
            f"{result['accepted_mean']:.1f}±{result['accepted_std']:.1f}"
        )

    print("=" * 100)
    logger.info(f"Comprehensive K-sweep completed. Results saved to {csv_file} and {json_file}")


if __name__ == "__main__":
    main()
