"""
Benchmarking Harness: Speculative Decoding vs Vanilla Decoding

Compares performance of speculative decoding against vanilla greedy decoding
under different configurations (model, k, batch size, etc.).

Usage:
    python -m scripts.benchmark_specdec_vs_vanilla \
        --model_name gpt2 \
        --device cuda \
        --impl torch \
        --batch_size 4 \
        --max_tokens 64 \
        --k 4 \
        --num_prompts 16 \
        --output_csv benchmarks.csv
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from specdec.benchmarks.core import run_single_benchmark, validate_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_summary(results: Dict[str, any]) -> None:
    """Print human-readable summary of benchmark results."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)
    logger.info(f"MODEL: {results['model_name']}")
    if (
        results.get("draft_model_name")
        and results["draft_model_name"] != results["model_name"]
    ):
        logger.info(f"DRAFT_MODEL: {results['draft_model_name']}")
    logger.info(f"DEVICE: {results['device']}")
    logger.info(f"IMPL: {results['impl']}")
    logger.info(f"BATCH_SIZE: {results['batch_size']}")
    logger.info(f"MAX_TOKENS: {results['max_tokens']}")
    logger.info(f"K: {results['k']}")
    logger.info("")
    logger.info(
        f"VANILLA:  tok/s = {results['vanilla_tok_s']:.2f},  "
        f"total_tokens = {results['total_tokens_vanilla']},  "
        f"total_time = {results['total_time_vanilla']:.2f}s"
    )
    logger.info(
        f"SPECDEC:  tok/s = {results['specdec_tok_s']:.2f},  "
        f"total_tokens = {results['total_tokens_specdec']},  "
        f"total_time = {results['total_time_specdec']:.2f}s"
    )
    logger.info(
        f"ACCEPTANCE_RATE = {results['acceptance_rate']:.3f},  "
        f"proposed = {results['total_proposed']}, accepted = {results['total_accepted']}"
    )
    logger.info(f"SPEEDUP = {results['speedup']:.2f}x")
    logger.info("=" * 80)


def write_csv(results: Dict[str, any], csv_path: str) -> None:
    """
    Write benchmark results to CSV file.

    Args:
        results: Results dictionary
        csv_path: Path to CSV file
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        fieldnames = [
            "model_name",
            "draft_model_name",
            "device",
            "impl",
            "batch_size",
            "max_tokens",
            "k",
            "num_prompts",
            "total_tokens_vanilla",
            "total_time_vanilla",
            "vanilla_tok_s",
            "total_tokens_specdec",
            "total_time_specdec",
            "specdec_tok_s",
            "acceptance_rate",
            "total_proposed",
            "total_accepted",
            "speedup",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)

    logger.info(f"Results written to {csv_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Speculative Decoding vs Vanilla Decoding"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Base model name (default: gpt2)",
    )
    parser.add_argument(
        "--draft_model_name",
        type=str,
        default=None,
        help="Draft model name (default: same as base model for perfect draft)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="torch",
        choices=["fake", "torch"],
        help="Implementation type (default: torch)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Maximum tokens per prompt (default: 64)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of draft tokens per step (default: 4)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=16,
        help="Total number of prompts to test (default: 16)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to CSV file for results (optional)",
    )

    args = parser.parse_args()

    try:
        # Validate device
        device = validate_device(args.device)
        if device != args.device and args.device != "auto":
            logger.warning(f"Device {args.device} not available, using {device}")

        # Run benchmark
        results = run_single_benchmark(
            model_name=args.model_name,
            draft_model_name=args.draft_model_name,
            device=device,
            impl=args.impl,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            k=args.k,
            num_prompts=args.num_prompts,
            verbose=True,
        )

        # Print summary
        print_summary(results)

        # Write CSV if requested
        if args.output_csv:
            write_csv(results, args.output_csv)

        logger.info("Benchmark completed successfully")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
