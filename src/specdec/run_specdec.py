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

from specdec.core.pipeline import SpeculativePipeline  # noqa: E402


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
        "--draft-mode",
        type=str,
        choices=["vanilla", "medusa", "eagle"],
        default="vanilla",
        help="Draft generation strategy (default: vanilla)",
    )
    parser.add_argument(
        "--force-device",
        type=str,
        choices=["cpu", "mps"],
        help="Force both models to same device (cpu or mps)",
    )

    # Policy arguments
    parser.add_argument(
        "--policy",
        type=str,
        choices=["longest_prefix", "conf_threshold", "topk_agree", "typical"],
        default="longest_prefix",
        help="Acceptance policy (default: longest_prefix)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        help="Confidence threshold for conf_threshold policy (default: 0.5)",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Top-k value for topk_agree policy (default: 5)",
    )
    parser.add_argument(
        "--p",
        type=float,
        help="Typical acceptance probability for typical policy (default: 0.9)",
    )

    # Controller arguments
    parser.add_argument(
        "--controller",
        type=str,
        choices=["fixed", "adaptive"],
        default="fixed",
        help="K controller type (default: fixed)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=4,
        help="Fixed K value for fixed controller (default: 4)",
    )
    parser.add_argument(
        "--adaptive-K",
        action="store_true",
        help="Use adaptive K controller (mutually exclusive with --K)",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        help="Minimum K for adaptive controller (default: 1)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        help="Maximum K for adaptive controller (default: 8)",
    )
    parser.add_argument(
        "--target-acceptance",
        type=float,
        help="Target acceptance rate for adaptive controller (default: 0.7)",
    )

    # Optimization and profiling arguments
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable comprehensive profiling and performance analysis",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        help="Directory to save profiling traces (default: profiles/)",
    )
    parser.add_argument(
        "--disable-optimization",
        action="store_true",
        help="Disable performance optimizations (mixed precision, etc.)",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Handle mutual exclusivity between --K and --adaptive-K
        # Check if both were explicitly provided by looking at sys.argv
        has_K = any(arg.startswith("--K") for arg in sys.argv)
        has_adaptive_K = "--adaptive-K" in sys.argv
        if has_K and has_adaptive_K:
            logger.error("Cannot specify both --K and --adaptive-K")
            sys.exit(1)

        # Determine controller type and parameters
        if args.adaptive_K:
            controller = "adaptive"
            controller_params = {
                "min_k": args.min_k,
                "max_k": args.max_k,
                "target_acceptance_rate": args.target_acceptance,
            }
        else:
            controller = "fixed"
            controller_params = {"k": args.K}

        # Determine policy parameters
        policy_params = {}
        if args.tau is not None:
            policy_params["tau"] = args.tau
        if args.k is not None:
            policy_params["k"] = args.k
        if args.p is not None:
            policy_params["p"] = args.p

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
            policy=args.policy,
            policy_params=policy_params,
            controller=controller,
            controller_params=controller_params,
            draft_mode=args.draft_mode,
            enable_optimization=not args.disable_optimization,
            enable_profiling=args.profile,
            profile_dir=args.profile_dir,
        )

        # Generate text
        logger.info(f"Generating text for prompt: '{args.prompt[:50]}...'")
        logger.info(f"max_tokens: {args.max_tokens}, type: {type(args.max_tokens)}")
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
            "draft_mode": result["draft_mode"],
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
