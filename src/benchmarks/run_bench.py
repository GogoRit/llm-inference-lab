"""
Benchmark Runner for LLM Inference Lab

Runs multiple inference iterations and measures performance metrics including
latency, throughput, and statistical analysis. Supports both local baseline
and HTTP vLLM server endpoints.

Usage:
    # Local baseline benchmarking
    python -m src.benchmarks.run_bench --prompt "Hello, world!" --iterations 5

    # HTTP server benchmarking
    python -m src.benchmarks.run_bench --prompt "Hello, world!" --mode http \\
        --host 127.0.0.1 --port 8000

    # Configuration-driven benchmarking
    python -m src.benchmarks.run_bench --config configs/baseline.yaml --iterations 10
"""

import argparse
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from server.local_baseline import LocalBaselineRunner  # noqa: E402
from server.ping_vllm import VLLMPingClient  # noqa: E402
from specdec.pipeline import SpeculativePipeline  # noqa: E402
from benchmarks.quality_eval import create_evaluator  # noqa: E402


class BenchmarkRunner:
    """Runs performance benchmarks on local baseline or HTTP vLLM servers."""

    def __init__(
        self,
        config_path: str = None,
        mode: str = "local",
        host: str = None,
        port: int = None,
        compare_baseline: bool = False,
        eval_perplexity: bool = False,
    ):
        """
        Initialize the benchmark runner.

        Args:
            config_path: Path to YAML configuration file
            mode: Benchmark mode ("local", "http", or "specdec")
            host: Server hostname (for HTTP mode)
            port: Server port (for HTTP mode)
            compare_baseline: Whether to run baseline comparison (for specdec mode)
            eval_perplexity: Whether to evaluate text quality using perplexity (HF mode only)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.mode = mode
        self.compare_baseline = compare_baseline

        if mode == "local":
            self.runner = LocalBaselineRunner(
                model_name=self.config.get("model", "facebook/opt-125m"),
                config_path=config_path,
            )
            self.runner_name = "LocalBaselineRunner"
        elif mode == "http":
            self.runner = VLLMPingClient(
                host=host or self.config.get("host", "127.0.0.1"),
                port=port or self.config.get("port", 8000),
                config_path=config_path,
            )
            self.runner_name = "VLLMPingClient"
        elif mode == "specdec":
            self.runner = SpeculativePipeline(
                config_path=config_path,
                base_model=self.config.get("base_model"),
                draft_model=self.config.get("draft_model"),
                max_draft=self.config.get("max_draft"),
                device=self.config.get("device"),
                seed=self.config.get("seed"),
            )
            self.runner_name = "SpeculativePipeline"

            # Initialize baseline runner for comparison if requested
            if compare_baseline:
                self.baseline_runner = LocalBaselineRunner(
                    model_name=self.config.get("base_model", "facebook/opt-125m"),
                    config_path=config_path,
                )
                self.baseline_runner_name = "LocalBaselineRunner"
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'local', 'http', or 'specdec'"
            )

        # Initialize perplexity evaluator if requested and in HF mode
        self.evaluator = None
        if eval_perplexity and mode == "specdec" and self.config.get("implementation") == "hf":
            try:
                eval_model = self.config.get("eval_model", "sshleifer/tiny-gpt2")
                eval_device = self.config.get("eval_device", "cpu")
                self.evaluator = create_evaluator(eval_model, eval_device)
                self.logger.info(f"Initialized perplexity evaluator with {eval_model} on {eval_device}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize perplexity evaluator: {e}")
                self.evaluator = None

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load benchmark configuration."""
        default_config = {
            "model": "facebook/opt-125m",
            "max_new_tokens": 48,
            "temperature": 0.7,
            "do_sample": True,
            "iterations": 5,
            "warmup_iterations": 1,
            "device_priority": ["mps", "cuda", "cpu"],
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded benchmark config from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                self.logger.info("Using default benchmark configuration")
        else:
            self.logger.info("Using default benchmark configuration")

        return default_config

    def run_benchmark(self, prompt: str, iterations: int = None) -> Dict[str, Any]:
        """
        Run benchmark with multiple iterations.

        Args:
            prompt: Input prompt for generation
            iterations: Number of benchmark iterations (uses config if None)

        Returns:
            Dictionary containing benchmark results and statistics
        """
        if iterations is None:
            iterations = self.config.get("iterations", 5)

        warmup_iterations = self.config.get("warmup_iterations", 1)

        self.logger.info(
            f"Starting {self.mode} benchmark: {iterations} iterations, "
            f"{warmup_iterations} warmup"
        )
        self.logger.info(f"Prompt: '{prompt}'")

        if self.mode == "local":
            self.logger.info(f"Device: {self.runner.device}")
        elif self.mode == "http":
            self.logger.info(f"Server: {self.runner.base_url}")
        elif self.mode == "specdec":
            self.logger.info(f"Implementation: {self.runner.implementation}")

        # Test connectivity for HTTP mode
        if self.mode == "http":
            if not self.runner.ping():
                self.logger.error("Server is not reachable. Cannot run benchmark.")
                return {"error": "Server not reachable"}

        # Warmup runs
        if warmup_iterations > 0:
            self.logger.info(f"Running {warmup_iterations} warmup iterations...")
            for i in range(warmup_iterations):
                if self.mode == "local":
                    self.runner.run(prompt, self.config.get("max_new_tokens", 48))
                else:
                    self.runner.generate(prompt, self.config.get("max_tokens", 50))

        # Benchmark runs
        latencies = []
        tokens_per_second = []
        acceptance_rates = []
        results = []

        self.logger.info("Starting benchmark iterations...")
        for i in range(iterations):
            if self.mode == "local":
                result = self.runner.run(prompt, self.config.get("max_new_tokens", 48))
                latency_ms = result["latency_ms"]
                tokens_generated = result["max_new_tokens"]
                acceptance_rate = None  # Not applicable for baseline
            elif self.mode == "http":
                result = self.runner.generate(prompt, self.config.get("max_tokens", 50))
                if not result["success"]:
                    self.logger.error(f"Iteration {i+1} failed: {result['error']}")
                    continue
                latency_ms = result["latency_ms"]
                tokens_generated = result["tokens_generated"]
                acceptance_rate = None  # Not applicable for HTTP
            elif self.mode == "specdec":
                result = self.runner.generate(
                    prompt,
                    max_tokens=self.config.get("max_new_tokens", 48),
                    temperature=self.config.get("temperature"),
                    do_sample=self.config.get("do_sample"),
                )
                latency_ms = result["latency_ms"]
                tokens_generated = len(result["generated_tokens"])
                acceptance_rate = result["acceptance_rate"]
                
                # Add perplexity evaluation if requested and in HF mode
                if hasattr(self, 'evaluator') and self.evaluator is not None:
                    perplexity_result = self.evaluator.calculate_perplexity(result["text"])
                    result["perplexity"] = perplexity_result["perplexity"]
                    result["perplexity_loss"] = perplexity_result["loss"]

            tokens_per_sec = tokens_generated / (latency_ms / 1000.0)

            latencies.append(latency_ms)
            tokens_per_second.append(tokens_per_sec)
            if acceptance_rate is not None:
                acceptance_rates.append(acceptance_rate)
            results.append(result)

            self.logger.debug(
                f"Iteration {i+1}: {latency_ms:.2f}ms, {tokens_per_sec:.2f} tokens/sec"
                + (
                    f", acceptance_rate={acceptance_rate:.3f}"
                    if acceptance_rate is not None
                    else ""
                )
            )

        # Calculate statistics
        stats = {
            "mode": self.mode,
            "runner_name": self.runner_name,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "prompt": prompt,
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "min": min(latencies),
                "max": max(latencies),
                "raw": latencies,
            },
            "tokens_per_second": {
                "mean": statistics.mean(tokens_per_second),
                "median": statistics.median(tokens_per_second),
                "std": (
                    statistics.stdev(tokens_per_second)
                    if len(tokens_per_second) > 1
                    else 0.0
                ),
                "min": min(tokens_per_second),
                "max": max(tokens_per_second),
                "raw": tokens_per_second,
            },
            "raw_results": results,
        }

        # Add acceptance rate statistics for speculative decoding
        if acceptance_rates:
            stats["acceptance_rate"] = {
                "mean": statistics.mean(acceptance_rates),
                "median": statistics.median(acceptance_rates),
                "std": (
                    statistics.stdev(acceptance_rates)
                    if len(acceptance_rates) > 1
                    else 0.0
                ),
                "min": min(acceptance_rates),
                "max": max(acceptance_rates),
                "raw": acceptance_rates,
            }

        # Add mode-specific information
        if self.mode == "local":
            stats["device"] = self.runner.device
            stats["model"] = self.runner.model_name
        elif self.mode == "http":
            stats["server"] = self.runner.base_url
            stats["model"] = self.config.get("model", "unknown")
        elif self.mode == "specdec":
            stats["device"] = self.runner.device
            stats["base_model"] = self.runner.config["base_model"]
            stats["draft_model"] = self.runner.config["draft_model"]
            stats["max_draft"] = self.runner.config["max_draft"]

        # Run baseline comparison if requested
        if self.mode == "specdec" and self.compare_baseline:
            self.logger.info("Running baseline comparison...")
            baseline_stats = self._run_baseline_comparison(prompt, iterations)
            stats["baseline_comparison"] = baseline_stats

        return stats

    def _run_baseline_comparison(self, prompt: str, iterations: int) -> Dict[str, Any]:
        """Run baseline comparison for speculative decoding."""
        baseline_latencies = []
        baseline_tokens_per_second = []

        for i in range(iterations):
            result = self.baseline_runner.run(
                prompt, self.config.get("max_new_tokens", 48)
            )
            latency_ms = result["latency_ms"]
            tokens_generated = result["max_new_tokens"]
            tokens_per_sec = tokens_generated / (latency_ms / 1000.0)

            baseline_latencies.append(latency_ms)
            baseline_tokens_per_second.append(tokens_per_sec)

        return {
            "latency_ms": {
                "mean": statistics.mean(baseline_latencies),
                "median": statistics.median(baseline_latencies),
                "std": (
                    statistics.stdev(baseline_latencies)
                    if len(baseline_latencies) > 1
                    else 0.0
                ),
                "min": min(baseline_latencies),
                "max": max(baseline_latencies),
            },
            "tokens_per_second": {
                "mean": statistics.mean(baseline_tokens_per_second),
                "median": statistics.median(baseline_tokens_per_second),
                "std": (
                    statistics.stdev(baseline_tokens_per_second)
                    if len(baseline_tokens_per_second) > 1
                    else 0.0
                ),
                "min": min(baseline_tokens_per_second),
                "max": max(baseline_tokens_per_second),
            },
        }

    def print_summary(self, stats: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        self.logger.info("=== Benchmark Summary ===")
        self.logger.info(f"Mode: {stats['mode']} ({stats['runner_name']})")

        if stats["mode"] == "local":
            self.logger.info(f"Model: {stats['model']}")
            self.logger.info(f"Device: {stats['device']}")
        elif stats["mode"] == "http":
            self.logger.info(f"Model: {stats['model']}")
            self.logger.info(f"Server: {stats['server']}")
        elif stats["mode"] == "specdec":
            self.logger.info(f"Base Model: {stats['base_model']}")
            self.logger.info(f"Draft Model: {stats['draft_model']}")
            self.logger.info(f"Max Draft: {stats['max_draft']}")
            self.logger.info(f"Device: {stats['device']}")

        self.logger.info(
            f"Iterations: {stats['iterations']} (warmup: {stats['warmup_iterations']})"
        )
        self.logger.info(f"Prompt: '{stats['prompt']}'")

        self.logger.info("\n--- Latency (ms) ---")
        self.logger.info(f"Mean:   {stats['latency_ms']['mean']:.2f}")
        self.logger.info(f"Median: {stats['latency_ms']['median']:.2f}")
        self.logger.info(f"Std:    {stats['latency_ms']['std']:.2f}")
        self.logger.info(f"Min:    {stats['latency_ms']['min']:.2f}")
        self.logger.info(f"Max:    {stats['latency_ms']['max']:.2f}")

        self.logger.info("\n--- Throughput (tokens/sec) ---")
        self.logger.info(f"Mean:   {stats['tokens_per_second']['mean']:.2f}")
        self.logger.info(f"Median: {stats['tokens_per_second']['median']:.2f}")
        self.logger.info(f"Std:    {stats['tokens_per_second']['std']:.2f}")
        self.logger.info(f"Min:    {stats['tokens_per_second']['min']:.2f}")
        self.logger.info(f"Max:    {stats['tokens_per_second']['max']:.2f}")

        # Print acceptance rate for speculative decoding
        if "acceptance_rate" in stats:
            self.logger.info("\n--- Acceptance Rate ---")
            self.logger.info(f"Mean:   {stats['acceptance_rate']['mean']:.3f}")
            self.logger.info(f"Median: {stats['acceptance_rate']['median']:.3f}")
            self.logger.info(f"Std:    {stats['acceptance_rate']['std']:.3f}")
            self.logger.info(f"Min:    {stats['acceptance_rate']['min']:.3f}")
            self.logger.info(f"Max:    {stats['acceptance_rate']['max']:.3f}")

        # Print baseline comparison if available
        if "baseline_comparison" in stats:
            self.logger.info("\n--- Baseline Comparison ---")
            baseline = stats["baseline_comparison"]

            # Calculate speedup
            specdec_latency = stats["latency_ms"]["mean"]
            baseline_latency = baseline["latency_ms"]["mean"]
            speedup = baseline_latency / specdec_latency if specdec_latency > 0 else 0.0

            self.logger.info(f"Speculative Decoding Latency: {specdec_latency:.2f}ms")
            self.logger.info(f"Baseline Latency: {baseline_latency:.2f}ms")
            self.logger.info(f"Speedup: {speedup:.2f}x")

            specdec_throughput = stats["tokens_per_second"]["mean"]
            baseline_throughput = baseline["tokens_per_second"]["mean"]
            throughput_improvement = (
                (specdec_throughput / baseline_throughput - 1) * 100
                if baseline_throughput > 0
                else 0.0
            )

            self.logger.info(
                f"Speculative Decoding Throughput: {specdec_throughput:.2f} tokens/sec"
            )
            self.logger.info(
                f"Baseline Throughput: {baseline_throughput:.2f} tokens/sec"
            )
            self.logger.info(f"Throughput Improvement: {throughput_improvement:+.1f}%")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Runner")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for generation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of benchmark iterations (uses config default if not specified)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "http", "specdec"],
        default="local",
        help="Benchmark mode: local baseline, HTTP server, or speculative decoding",
    )
    parser.add_argument(
        "--host", type=str, default=None, help="Server hostname (for HTTP mode)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Server port (for HTTP mode)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run baseline comparison (for specdec mode only)",
    )
    parser.add_argument(
        "--eval-perplexity",
        action="store_true",
        help="Evaluate text quality using perplexity (HF mode only)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run benchmark
    benchmark = BenchmarkRunner(
        config_path=args.config,
        mode=args.mode,
        host=args.host,
        port=args.port,
        compare_baseline=args.compare_baseline,
        eval_perplexity=args.eval_perplexity,
    )
    stats = benchmark.run_benchmark(args.prompt, args.iterations)

    if "error" in stats:
        benchmark.logger.error(f"Benchmark failed: {stats['error']}")
    else:
        benchmark.print_summary(stats)


if __name__ == "__main__":
    main()
