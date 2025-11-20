"""
Results handling for K-sweep benchmarking.

Provides functions for saving results to CSV and JSON files.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def save_results(
    results: List[Dict],
    detailed_results: List[Dict],
    system_info: Dict[str, Any],
    output_dir: str,
    device: str,
) -> Tuple[Path, Path]:
    """
    Save results to CSV and JSON files.

    Args:
        results: Summary results list
        detailed_results: Detailed per-iteration results
        system_info: System and environment metadata
        output_dir: Output directory path
        device: Device name (cuda, mps, cpu)

    Returns:
        Tuple of (csv_file_path, json_file_path)
    """
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
