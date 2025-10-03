"""
CLI Entrypoint for Speculative Decoding

Command-line interface for running speculative decoding with JSON output.
Supports configuration files and various generation parameters.

Usage:
    python -m src.specdec.run_specdec --prompt "Explain KV cache simply." \\
        --max-tokens 64 --verbose
    python -m src.specdec.run_specdec --config configs/specdec.yaml \\
        --prompt "Test" --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# No additional imports needed

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from specdec.pipeline import SpeculativePipeline  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speculative Decoding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.specdec.run_specdec --prompt "Hello world" \\
      --max-tokens 32
  python -m src.specdec.run_specdec --prompt "Hello world" \\
      --max-tokens 32
  python -m src.specdec.run_specdec --config configs/specdec.yaml \\
      --prompt "Test" --verbose
        """,
    )

    # Required arguments
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")

    # Optional arguments
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate (overrides config)"
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Model parameters
    parser.add_argument(
        "--base-model", type=str, help="Base model name (overrides config)"
    )
    parser.add_argument(
        "--draft-model", type=str, help="Draft model name (overrides config)"
    )
    parser.add_argument(
        "--max-draft", type=int, help="Maximum draft tokens per step (overrides config)"
    )
    parser.add_argument(
        "--temperature", type=float, help="Sampling temperature (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to run on (overrides config)",
    )
    parser.add_argument(
        "--impl",
        type=str,
        choices=["fake", "hf"],
        default="fake",
        help="Implementation type: fake (default) for testing, hf for real models",
    )
    parser.add_argument(
        "--force-device",
        type=str,
        choices=["cpu", "mps"],
        help="Force both models to same device (cpu or mps)",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize pipeline
        logger.info(f"Initializing speculative decoding pipeline (impl={args.impl})...")
        pipeline = SpeculativePipeline(
            config_path=args.config,
            base_model=args.base_model,
            draft_model=args.draft_model,
            max_draft=args.max_draft,
            device=args.device,
            seed=args.seed,
            implementation=args.impl,
            force_device=args.force_device,
        )

        # Generate text
        logger.info(f"Generating text for prompt: '{args.prompt[:50]}...'")
        result = pipeline.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Print JSON result to stdout
        output = {
            "latency_ms": result["latency_ms"],
            "proposed": result["proposed"],
            "accepted": result["accepted"],
            "acceptance_rate": result["acceptance_rate"],
            "tokens_per_sec": result["tokens_per_sec"],
            "text": result["text"],
            "impl": result["impl"],
            "device": result["device"],
            "base_model": result["base_model"],
            "draft_model": result["draft_model"],
            "dtype": result["dtype"],
        }

        print(json.dumps(output, indent=None))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
