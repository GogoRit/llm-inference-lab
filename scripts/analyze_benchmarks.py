"""
Benchmark Analysis Script

Analyzes aggregated benchmark CSV files to identify:
- Speedup vs k at fixed batch size
- Speedup vs batch size at fixed k
- Configs where speedup > 1.0 (SpecDec wins)
- Acceptance rate vs speedup relationships

Usage:
    python -m scripts.analyze_benchmarks \
        --aggregated_csv t4_gpt2_grid_aggregated.csv \
        --device cuda \
        --model_name gpt2
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import pandas for better CSV handling, fall back to csv module
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_csv_pandas(csv_path: str) -> List[Dict[str, any]]:
    """Load CSV using pandas."""
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def load_csv_stdlib(csv_path: str) -> List[Dict[str, any]]:
    """Load CSV using standard library csv module."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_aggregated_csv(csv_path: str) -> List[Dict[str, any]]:
    """
    Load aggregated CSV file.

    Args:
        csv_path: Path to aggregated CSV file

    Returns:
        List of dictionaries with benchmark results
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if HAS_PANDAS:
        logger.debug("Using pandas to load CSV")
        return load_csv_pandas(csv_path)
    else:
        logger.debug("Using standard library csv module")
        return load_csv_stdlib(csv_path)


def filter_results(
    results: List[Dict[str, any]],
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, any]]:
    """
    Filter results by device and/or model_name.

    Args:
        results: List of result dictionaries
        device: Optional device filter
        model_name: Optional model name filter

    Returns:
        Filtered list of results
    """
    filtered = results
    if device:
        filtered = [r for r in filtered if r.get("device") == device]
    if model_name:
        filtered = [r for r in filtered if r.get("model_name") == model_name]
    return filtered


def get_unique_configs(results: List[Dict[str, any]]) -> List[Tuple[str, ...]]:
    """
    Get unique (model_name, device, impl, max_tokens, num_prompts) combinations.

    Args:
        results: List of result dictionaries

    Returns:
        List of unique config tuples
    """
    configs = set()
    for r in results:
        key = (
            r.get("model_name", "unknown"),
            r.get("device", "unknown"),
            r.get("impl", "unknown"),
            str(r.get("max_tokens", "unknown")),
            str(r.get("num_prompts", "unknown")),
        )
        configs.add(key)
    return sorted(list(configs))


def convert_numeric(value: any, default: float = 0.0) -> float:
    """Convert value to float, handling strings and missing values."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return float(value)


def print_config_summary(results: List[Dict[str, any]]) -> None:
    """Print high-level summary of configurations found."""
    configs = get_unique_configs(results)
    logger.info("=" * 80)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Found {len(configs)} unique configuration(s):")
    for i, (model, device, impl, max_tokens, num_prompts) in enumerate(configs, 1):
        logger.info(
            f"  {i}. model={model}, device={device}, impl={impl}, "
            f"max_tokens={max_tokens}, num_prompts={num_prompts}"
        )
    logger.info("")


def print_config_table(
    results: List[Dict[str, any]],
    model_name: str,
    device: str,
    impl: str,
    max_tokens: str,
    num_prompts: str,
) -> None:
    """
    Print a table for a specific configuration.

    Args:
        results: Filtered results for this config
        model_name: Model name
        device: Device
        impl: Implementation
        max_tokens: Max tokens
        num_prompts: Number of prompts
    """
    # Filter to this exact config
    config_results = [
        r
        for r in results
        if (
            r.get("model_name") == model_name
            and r.get("device") == device
            and r.get("impl") == impl
            and str(r.get("max_tokens")) == max_tokens
            and str(r.get("num_prompts")) == num_prompts
        )
    ]

    if not config_results:
        return

    # Sort by batch_size, then k
    config_results.sort(
        key=lambda x: (
            convert_numeric(x.get("batch_size")),
            convert_numeric(x.get("k")),
        )
    )

    # Print header
    draft_model = config_results[0].get("draft_model_name", "unknown")
    logger.info("=" * 80)
    logger.info(
        f"CONFIG: {model_name} (draft: {draft_model}), device={device}, "
        f"max_tokens={max_tokens}, num_prompts={num_prompts}"
    )
    logger.info("=" * 80)
    logger.info("")
    logger.info(
        f"{'batch_size':<12} {'k':<6} {'vanilla_tok/s':<15} {'specdec_tok/s':<15} "
        f"{'speedup':<10} {'acc_rate':<10} {'*':<3}"
    )
    logger.info("-" * 80)

    best_speedup = -1.0
    best_config = None

    for r in config_results:
        batch_size = r.get("batch_size", "?")
        k = r.get("k", "?")
        vanilla_tok_s = convert_numeric(r.get("vanilla_tok_s"))
        specdec_tok_s = convert_numeric(r.get("specdec_tok_s"))
        speedup = convert_numeric(r.get("speedup"))
        acc_rate = convert_numeric(r.get("acceptance_rate"))

        # Track best speedup
        if speedup > best_speedup:
            best_speedup = speedup
            best_config = (batch_size, k)

        # Mark if speedup > 1.0
        marker = "*" if speedup > 1.0 else ""

        logger.info(
            f"{batch_size:<12} {k:<6} {vanilla_tok_s:<15.2f} {specdec_tok_s:<15.2f} "
            f"{speedup:<10.2f} {acc_rate:<10.3f} {marker:<3}"
        )

    logger.info("")
    logger.info(
        f"Best speedup: {best_speedup:.2f}x at batch_size={best_config[0]}, k={best_config[1]}"
    )
    logger.info("")
    logger.info("Legend: * = speedup > 1.0 (SpecDec faster than vanilla)")
    logger.info("")


def analyze_speedup_patterns(results: List[Dict[str, any]]) -> None:
    """
    Analyze speedup patterns across k and batch_size.

    Args:
        results: List of result dictionaries
    """
    logger.info("=" * 80)
    logger.info("SPEEDUP PATTERN ANALYSIS")
    logger.info("=" * 80)

    # Group by (model_name, device)
    groups = defaultdict(list)
    for r in results:
        key = (r.get("model_name"), r.get("device"))
        groups[key].append(r)

    for (model_name, device), group_results in sorted(groups.items()):
        logger.info("")
        logger.info(f"Model: {model_name}, Device: {device}")

        # Find configs with speedup > 1.0
        winning_configs = [
            r for r in group_results if convert_numeric(r.get("speedup")) > 1.0
        ]

        if winning_configs:
            logger.info(
                f"  ✓ Found {len(winning_configs)} config(s) where SpecDec wins (speedup > 1.0):"
            )
            for r in winning_configs[:5]:  # Show top 5
                logger.info(
                    f"    batch={r.get('batch_size')}, k={r.get('k')}, "
                    f"speedup={convert_numeric(r.get('speedup')):.2f}x, "
                    f"acc_rate={convert_numeric(r.get('acceptance_rate')):.3f}"
                )
        else:
            logger.info("  ✗ No configs found where SpecDec wins (all speedup <= 1.0)")

        # Analyze speedup vs k (at fixed batch_size)
        batch_sizes = sorted(
            set(convert_numeric(r.get("batch_size")) for r in group_results)
        )
        logger.info("")
        logger.info("  Speedup vs k (at fixed batch_size):")
        for bs in batch_sizes[:3]:  # Show first 3 batch sizes
            bs_results = [
                r for r in group_results if convert_numeric(r.get("batch_size")) == bs
            ]
            bs_results.sort(key=lambda x: convert_numeric(x.get("k")))
            speedups = [convert_numeric(r.get("speedup")) for r in bs_results]
            k_values = [r.get("k") for r in bs_results]
            logger.info(
                f"    batch_size={int(bs)}: k={k_values} -> speedup={[f'{s:.2f}' for s in speedups]}"
            )

        # Analyze speedup vs batch_size (at fixed k)
        k_values = sorted(set(convert_numeric(r.get("k")) for r in group_results))
        logger.info("")
        logger.info("  Speedup vs batch_size (at fixed k):")
        for k in k_values[:3]:  # Show first 3 k values
            k_results = [r for r in group_results if convert_numeric(r.get("k")) == k]
            k_results.sort(key=lambda x: convert_numeric(x.get("batch_size")))
            speedups = [convert_numeric(r.get("speedup")) for r in k_results]
            batch_sizes = [r.get("batch_size") for r in k_results]
            logger.info(
                f"    k={int(k)}: batch_size={batch_sizes} -> speedup={[f'{s:.2f}' for s in speedups]}"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze aggregated benchmark CSV files"
    )
    parser.add_argument(
        "--aggregated_csv",
        type=str,
        required=True,
        help="Path to aggregated CSV file (required)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Filter by device (optional)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Filter by model name (optional)",
    )

    args = parser.parse_args()

    try:
        # Load CSV
        logger.info(f"Loading aggregated CSV: {args.aggregated_csv}")
        results = load_aggregated_csv(args.aggregated_csv)
        logger.info(f"Loaded {len(results)} result rows")

        # Filter if requested
        if args.device or args.model_name:
            results = filter_results(
                results, device=args.device, model_name=args.model_name
            )
            logger.info(f"After filtering: {len(results)} result rows")

        if not results:
            logger.error("No results found after filtering. Exiting.")
            sys.exit(1)

        # Print summary
        print_config_summary(results)

        # Print tables for each unique config
        configs = get_unique_configs(results)
        for config in configs:
            model_name, device, impl, max_tokens, num_prompts = config
            print_config_table(
                results, model_name, device, impl, max_tokens, num_prompts
            )

        # Analyze patterns
        analyze_speedup_patterns(results)

        logger.info("")
        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
