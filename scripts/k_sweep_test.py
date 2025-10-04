#!/usr/bin/env python3
"""
K-Sweep Testing Script for Speculative Decoding
Tests acceptance rates and performance for K = 1-4 with compatible models.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec.pipeline import SpeculativePipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_k_sweep(
    base_model="gpt2",
    draft_model="distilgpt2",
    prompt="Explain KV cache simply.",
    max_tokens=32,
    iterations=3,
):
    """Run K-sweep test for K=1 to K=4."""

    results = []

    for k in range(1, 5):  # K = 1, 2, 3, 4
        logger.info(f"Testing K={k}...")

        k_results = []

        for i in range(iterations):
            logger.info(f"  Iteration {i+1}/{iterations}")

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

                k_results.append(
                    {
                        "iteration": i + 1,
                        "latency_ms": latency_ms,
                        "tokens_per_sec": tokens_per_sec,
                        "acceptance_rate": acceptance_rate,
                        "proposed": proposed,
                        "accepted": accepted,
                        "text": (
                            result.get("text", "")[:100] + "..."
                            if len(result.get("text", "")) > 100
                            else result.get("text", "")
                        ),
                    }
                )

                logger.info(
                    f"    K={k}, Iter={i+1}: {tokens_per_sec:.2f} tok/s, "
                    f"{acceptance_rate:.3f} accept rate"
                )

            except Exception as e:
                logger.error(f"    K={k}, Iter={i+1} failed: {e}")
                k_results.append({"iteration": i + 1, "error": str(e)})

        # Calculate averages for this K
        valid_results = [r for r in k_results if "error" not in r]
        if valid_results:
            avg_latency = sum(r["latency_ms"] for r in valid_results) / len(valid_results)
            avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in valid_results) / len(
                valid_results
            )
            avg_acceptance_rate = sum(r["acceptance_rate"] for r in valid_results) / len(
                valid_results
            )
            avg_proposed = sum(r["proposed"] for r in valid_results) / len(valid_results)
            avg_accepted = sum(r["accepted"] for r in valid_results) / len(valid_results)

            results.append(
                {
                    "k": k,
                    "avg_latency_ms": avg_latency,
                    "avg_tokens_per_sec": avg_tokens_per_sec,
                    "avg_acceptance_rate": avg_acceptance_rate,
                    "avg_proposed": avg_proposed,
                    "avg_accepted": avg_accepted,
                    "iterations": len(valid_results),
                    "raw_results": k_results,
                }
            )

            logger.info(
                f"  K={k} AVERAGE: {avg_tokens_per_sec:.2f} tok/s, "
                f"{avg_acceptance_rate:.3f} accept rate"
            )
        else:
            logger.error(f"  K={k}: All iterations failed")

    return results


def main():
    parser = argparse.ArgumentParser(description="K-Sweep Testing for Speculative Decoding")
    parser.add_argument("--base-model", default="gpt2", help="Base model name")
    parser.add_argument("--draft-model", default="distilgpt2", help="Draft model name")
    parser.add_argument("--prompt", default="Explain KV cache simply.", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per K")
    parser.add_argument("--output", default="k_sweep_results.json", help="Output file for results")

    args = parser.parse_args()

    logger.info(f"Starting K-sweep test: {args.base_model} + {args.draft_model}")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Max tokens: {args.max_tokens}, Iterations: {args.iterations}")

    results = run_k_sweep(
        base_model=args.base_model,
        draft_model=args.draft_model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        iterations=args.iterations,
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("K-SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'K':<3} {'Latency (ms)':<12} {'Tokens/sec':<12} {'Accept Rate':<12} "
        f"{'Proposed':<10} {'Accepted':<10}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['k']:<3} {result['avg_latency_ms']:<12.1f} "
            f"{result['avg_tokens_per_sec']:<12.2f} {result['avg_acceptance_rate']:<12.3f} "
            f"{result['avg_proposed']:<10.1f} {result['avg_accepted']:<10.1f}"
        )

    print("=" * 80)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
