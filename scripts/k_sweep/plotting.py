"""
Plotting utilities for K-sweep benchmarking.

Provides functions for creating performance visualization plots.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def create_plots(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create performance plots for K-sweep results.

    Args:
        results: Summary results list with K values and metrics
        output_dir: Output directory path
    """
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
