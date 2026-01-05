"""
Grid Runner for Speculative Decoding Benchmarks

Sweeps over multiple configurations (models, k, batch sizes) and generates
consolidated CSV outputs and optional Markdown reports.

Usage:
    python -m scripts.run_benchmark_grid \
        --device cuda \
        --impl torch \
        --output_csv t4_gpt2_grid.csv \
        --repeats 3 \
        --output_md benchmarks.md
"""

import argparse
import csv
import logging
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from specdec.benchmarks.core import run_single_benchmark, validate_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Experiment Matrix Configuration
# Define different experiment sets for different devices/models

# GPT-2 experiments (for T4/MPS)
GPT2_EXPERIMENTS = {
    "model_configs": [
        {
            "model_name": "gpt2",
            "draft_model_name": None,
        },  # Perfect draft (gpt2 -> gpt2)
        {"model_name": "gpt2", "draft_model_name": "distilgpt2"},  # Draft -> target
    ],
    "batch_sizes": [1, 2, 4, 8],
    "k_values": [1, 2, 4, 8],
    "max_tokens": 64,
    "num_prompts": 32,
}

# Tiny model experiments (for CPU sanity checks)
TINY_EXPERIMENTS = {
    "model_configs": [
        {"model_name": "sshleifer/tiny-gpt2", "draft_model_name": None},
    ],
    "batch_sizes": [1, 2, 4],
    "k_values": [1, 2, 4],
    "max_tokens": 64,
    "num_prompts": 32,
}

# Default configuration grid (uses GPT-2 experiments)
MODEL_CONFIGS = GPT2_EXPERIMENTS["model_configs"]
BATCH_SIZES = GPT2_EXPERIMENTS["batch_sizes"]
K_VALUES = GPT2_EXPERIMENTS["k_values"]
MAX_TOKENS = GPT2_EXPERIMENTS["max_tokens"]
NUM_PROMPTS = GPT2_EXPERIMENTS["num_prompts"]


def run_grid(
    device: str,
    impl: str,
    output_csv: str,
    repeats: int = 1,
    model_configs: Optional[List[Dict[str, Optional[str]]]] = None,
    batch_sizes: Optional[List[int]] = None,
    k_values: Optional[List[int]] = None,
    max_tokens: Optional[int] = None,
    num_prompts: Optional[int] = None,
) -> List[Dict[str, any]]:
    """
    Run benchmark grid and return all results.

    Args:
        device: Device to run on
        impl: Implementation type
        output_csv: Path to output CSV file
        repeats: Number of repeats per config
        model_configs: List of model configs (defaults to MODEL_CONFIGS)
        batch_sizes: List of batch sizes (defaults to BATCH_SIZES)
        k_values: List of k values (defaults to K_VALUES)
        max_tokens: Max tokens per prompt (defaults to MAX_TOKENS)
        num_prompts: Number of prompts (defaults to NUM_PROMPTS)

    Returns:
        List of all benchmark result dictionaries
    """
    model_configs = model_configs or MODEL_CONFIGS
    batch_sizes = batch_sizes or BATCH_SIZES
    k_values = k_values or K_VALUES
    max_tokens = max_tokens or MAX_TOKENS
    num_prompts = num_prompts or NUM_PROMPTS

    # Validate device
    device = validate_device(device)
    logger.info(f"Running grid on device: {device}")

    results = []
    total_configs = len(model_configs) * len(batch_sizes) * len(k_values) * repeats
    config_idx = 0

    for model_cfg in model_configs:
        model_name = model_cfg["model_name"]
        draft_model_name = model_cfg.get("draft_model_name")

        for batch_size in batch_sizes:
            for k in k_values:
                for repeat in range(repeats):
                    config_idx += 1
                    logger.info(
                        f"[{config_idx}/{total_configs}] Running benchmark: "
                        f"model={model_name}, batch={batch_size}, k={k}, repeat={repeat}..."
                    )

                    try:
                        metrics = run_single_benchmark(
                            model_name=model_name,
                            draft_model_name=draft_model_name,
                            device=device,
                            impl=impl,
                            batch_size=batch_size,
                            max_tokens=max_tokens,
                            k=k,
                            num_prompts=num_prompts,
                            verbose=False,  # Less verbose for grid runs
                        )
                        metrics["repeat"] = repeat
                        results.append(metrics)

                        logger.info(
                            f"  ✓ Speedup: {metrics['speedup']:.2f}x, "
                            f"acceptance_rate: {metrics['acceptance_rate']:.3f}"
                        )

                    except Exception as e:
                        logger.error(
                            f"  ✗ Failed: {e}",
                            exc_info=False,  # Don't print full traceback for grid runs
                        )
                        # Continue with next config

    return results


def aggregate_results(results: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Aggregate results by grouping configs and computing means/std devs.

    Args:
        results: List of raw benchmark results

    Returns:
        List of aggregated results with mean and std dev fields
    """
    # Group by config (excluding repeat)
    groups = defaultdict(list)
    for r in results:
        key = (
            r["model_name"],
            r["draft_model_name"],
            r["device"],
            r["impl"],
            r["batch_size"],
            r["max_tokens"],
            r["k"],
            r["num_prompts"],
        )
        groups[key].append(r)

    aggregated = []
    for key, group_results in groups.items():
        (
            model_name,
            draft_model_name,
            device,
            impl,
            batch_size,
            max_tokens,
            k,
            num_prompts,
        ) = key

        # Compute means
        mean_vanilla_tok_s = statistics.mean(
            [r["vanilla_tok_s"] for r in group_results]
        )
        mean_specdec_tok_s = statistics.mean(
            [r["specdec_tok_s"] for r in group_results]
        )
        mean_speedup = statistics.mean([r["speedup"] for r in group_results])
        mean_acceptance_rate = statistics.mean(
            [r["acceptance_rate"] for r in group_results]
        )
        mean_total_time_vanilla = statistics.mean(
            [r["total_time_vanilla"] for r in group_results]
        )
        mean_total_time_specdec = statistics.mean(
            [r["total_time_specdec"] for r in group_results]
        )

        # Compute std devs (if multiple repeats)
        if len(group_results) > 1:
            std_vanilla_tok_s = statistics.stdev(
                [r["vanilla_tok_s"] for r in group_results]
            )
            std_specdec_tok_s = statistics.stdev(
                [r["specdec_tok_s"] for r in group_results]
            )
            std_speedup = statistics.stdev([r["speedup"] for r in group_results])
            std_acceptance_rate = statistics.stdev(
                [r["acceptance_rate"] for r in group_results]
            )
        else:
            std_vanilla_tok_s = 0.0
            std_specdec_tok_s = 0.0
            std_speedup = 0.0
            std_acceptance_rate = 0.0

        # Use values from first result for non-aggregated fields
        first = group_results[0]

        aggregated.append(
            {
                "model_name": model_name,
                "draft_model_name": draft_model_name,
                "device": device,
                "impl": impl,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "k": k,
                "num_prompts": num_prompts,
                "repeats": len(group_results),
                "vanilla_tok_s": mean_vanilla_tok_s,
                "vanilla_tok_s_std": std_vanilla_tok_s,
                "specdec_tok_s": mean_specdec_tok_s,
                "specdec_tok_s_std": std_specdec_tok_s,
                "speedup": mean_speedup,
                "speedup_std": std_speedup,
                "acceptance_rate": mean_acceptance_rate,
                "acceptance_rate_std": std_acceptance_rate,
                "total_time_vanilla": mean_total_time_vanilla,
                "total_time_specdec": mean_total_time_specdec,
                # Use totals from first run (they should be similar across repeats)
                "total_tokens_vanilla": first["total_tokens_vanilla"],
                "total_tokens_specdec": first["total_tokens_specdec"],
                "total_proposed": first["total_proposed"],
                "total_accepted": first["total_accepted"],
            }
        )

    return aggregated


def write_csv(
    results: List[Dict[str, any]], csv_path: str, fieldnames: List[str]
) -> None:
    """
    Write results to CSV file.

    Args:
        results: List of result dictionaries
        csv_path: Path to CSV file
        fieldnames: List of field names for CSV columns
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Wrote {len(results)} rows to {csv_path}")


def generate_markdown_report(
    aggregated_results: List[Dict[str, any]],
    device: str,
    impl: str,
    output_md: str,
) -> None:
    """
    Generate a simple Markdown report with tables.

    Args:
        aggregated_results: List of aggregated benchmark results
        device: Device used
        impl: Implementation used
        output_md: Path to output Markdown file
    """
    # Group by (model_name, device, impl, max_tokens, num_prompts)
    groups = defaultdict(list)
    for r in aggregated_results:
        key = (
            r["model_name"],
            r["draft_model_name"],
            r["device"],
            r["impl"],
            r["max_tokens"],
            r["num_prompts"],
        )
        groups[key].append(r)

    with open(output_md, "w") as f:
        f.write("# SpecDec Benchmarks\n\n")
        f.write(f"Device: {device}  \n")
        f.write(f"Impl: {impl}  \n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")

        for key, group_results in sorted(groups.items()):
            (
                model_name,
                draft_model_name,
                _device,
                _impl,
                max_tokens,
                num_prompts,
            ) = key

            f.write(f"## {model_name}")
            if draft_model_name and draft_model_name != model_name:
                f.write(f" (draft: {draft_model_name})")
            f.write(f", max_tokens={max_tokens}, num_prompts={num_prompts}\n\n")

            # Sort by batch_size, then k
            group_results.sort(key=lambda x: (x["batch_size"], x["k"]))

            # Build table
            f.write(
                "| batch_size | k | vanilla_tok/s | specdec_tok/s | speedup | acceptance_rate |\n"
            )
            f.write(
                "|-----------:|---|--------------:|--------------:|--------:|----------------:|\n"
            )

            for r in group_results:
                vanilla_str = f"{r['vanilla_tok_s']:.1f}"
                if r.get("vanilla_tok_s_std", 0) > 0:
                    vanilla_str += f" ± {r['vanilla_tok_s_std']:.1f}"

                specdec_str = f"{r['specdec_tok_s']:.1f}"
                if r.get("specdec_tok_s_std", 0) > 0:
                    specdec_str += f" ± {r['specdec_tok_s_std']:.1f}"

                speedup_str = f"{r['speedup']:.2f}"
                if r.get("speedup_std", 0) > 0:
                    speedup_str += f" ± {r['speedup_std']:.2f}"

                acc_str = f"{r['acceptance_rate']:.3f}"
                if r.get("acceptance_rate_std", 0) > 0:
                    acc_str += f" ± {r['acceptance_rate_std']:.3f}"

                f.write(
                    f"| {r['batch_size']} | {r['k']} | {vanilla_str} | {specdec_str} | {speedup_str} | {acc_str} |\n"
                )

            f.write("\n")

    logger.info(f"Markdown report written to {output_md}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmark grid over multiple configurations"
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
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file (required)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per config (default: 1)",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Path to output Markdown report (optional)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help=f"Max tokens per prompt (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help=f"Number of prompts (default: {NUM_PROMPTS})",
    )
    parser.add_argument(
        "--experiment_set",
        type=str,
        default="gpt2",
        choices=["gpt2", "tiny"],
        help="Experiment set to use: 'gpt2' for full experiments, 'tiny' for CPU sanity checks (default: gpt2)",
    )

    args = parser.parse_args()

    try:
        # Validate device
        device = validate_device(args.device)
        if device != args.device and args.device != "auto":
            logger.warning(f"Device {args.device} not available, using {device}")

        # Select experiment set
        if args.experiment_set == "gpt2":
            exp_config = GPT2_EXPERIMENTS
            logger.info("Using GPT-2 experiment set")
        elif args.experiment_set == "tiny":
            exp_config = TINY_EXPERIMENTS
            logger.info("Using tiny model experiment set (for CPU sanity checks)")
        else:
            exp_config = GPT2_EXPERIMENTS

        # Run grid
        logger.info("Starting benchmark grid...")
        raw_results = run_grid(
            device=device,
            impl=args.impl,
            output_csv=args.output_csv,
            repeats=args.repeats,
            model_configs=exp_config["model_configs"],
            batch_sizes=exp_config["batch_sizes"] if args.max_tokens is None else None,
            k_values=exp_config["k_values"] if args.max_tokens is None else None,
            max_tokens=args.max_tokens or exp_config["max_tokens"],
            num_prompts=args.num_prompts or exp_config["num_prompts"],
        )

        if not raw_results:
            logger.error("No results collected. Exiting.")
            sys.exit(1)

        # Write raw CSV
        raw_fieldnames = [
            "model_name",
            "draft_model_name",
            "device",
            "impl",
            "batch_size",
            "max_tokens",
            "k",
            "num_prompts",
            "repeat",
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
        write_csv(raw_results, args.output_csv, raw_fieldnames)

        # Aggregate results
        logger.info("Aggregating results...")
        aggregated_results = aggregate_results(raw_results)

        # Write aggregated CSV
        aggregated_csv = args.output_csv.replace(".csv", "_aggregated.csv")
        aggregated_fieldnames = [
            "model_name",
            "draft_model_name",
            "device",
            "impl",
            "batch_size",
            "max_tokens",
            "k",
            "num_prompts",
            "repeats",
            "vanilla_tok_s",
            "vanilla_tok_s_std",
            "specdec_tok_s",
            "specdec_tok_s_std",
            "speedup",
            "speedup_std",
            "acceptance_rate",
            "acceptance_rate_std",
            "total_time_vanilla",
            "total_time_specdec",
            "total_tokens_vanilla",
            "total_tokens_specdec",
            "total_proposed",
            "total_accepted",
        ]
        write_csv(aggregated_results, aggregated_csv, aggregated_fieldnames)

        # Generate Markdown report if requested
        if args.output_md:
            logger.info("Generating Markdown report...")
            generate_markdown_report(
                aggregated_results, device, args.impl, args.output_md
            )

        logger.info("Grid benchmark completed successfully")
        logger.info(f"  Raw results: {args.output_csv}")
        logger.info(f"  Aggregated results: {aggregated_csv}")
        if args.output_md:
            logger.info(f"  Markdown report: {args.output_md}")

    except Exception as e:
        logger.error(f"Grid benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
