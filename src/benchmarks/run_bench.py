"""
Benchmark Runner for LLM Inference Lab

Runs multiple inference iterations and measures performance metrics including
latency, throughput, and statistical analysis.

Usage:
    python -m src.benchmarks.run_bench --prompt "Hello, world!" --iterations 5
    python -m src.benchmarks.run_bench --config configs/baseline.yaml --iterations 10
"""

import argparse
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add src to path for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from server.local_baseline import LocalBaselineRunner


class BenchmarkRunner:
    """Runs performance benchmarks on the local baseline runner."""

    def __init__(self, config_path: str = None):
        """
        Initialize the benchmark runner.

        Args:
            config_path: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.runner = LocalBaselineRunner(
            model_name=self.config.get("model", "facebook/opt-125m"),
            config_path=config_path
        )

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load benchmark configuration."""
        default_config = {
            "model": "facebook/opt-125m",
            "max_new_tokens": 48,
            "temperature": 0.7,
            "do_sample": True,
            "iterations": 5,
            "warmup_iterations": 1,
            "device_priority": ["mps", "cuda", "cpu"]
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
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
        
        self.logger.info(f"Starting benchmark: {iterations} iterations, {warmup_iterations} warmup")
        self.logger.info(f"Prompt: '{prompt}'")
        self.logger.info(f"Device: {self.runner.device}")
        
        # Warmup runs
        if warmup_iterations > 0:
            self.logger.info(f"Running {warmup_iterations} warmup iterations...")
            for i in range(warmup_iterations):
                self.runner.run(prompt, self.config.get("max_new_tokens", 48))
        
        # Benchmark runs
        latencies = []
        tokens_per_second = []
        results = []
        
        self.logger.info("Starting benchmark iterations...")
        for i in range(iterations):
            start_time = time.time()
            result = self.runner.run(prompt, self.config.get("max_new_tokens", 48))
            end_time = time.time()
            
            latency_ms = result["latency_ms"]
            tokens_generated = result["max_new_tokens"]
            tokens_per_sec = tokens_generated / (latency_ms / 1000.0)
            
            latencies.append(latency_ms)
            tokens_per_second.append(tokens_per_sec)
            results.append(result)
            
            self.logger.debug(f"Iteration {i+1}: {latency_ms:.2f}ms, {tokens_per_sec:.2f} tokens/sec")
        
        # Calculate statistics
        stats = {
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "prompt": prompt,
            "device": self.runner.device,
            "model": self.runner.model_name,
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "min": min(latencies),
                "max": max(latencies),
                "raw": latencies
            },
            "tokens_per_second": {
                "mean": statistics.mean(tokens_per_second),
                "median": statistics.median(tokens_per_second),
                "std": statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0.0,
                "min": min(tokens_per_second),
                "max": max(tokens_per_second),
                "raw": tokens_per_second
            },
            "raw_results": results
        }
        
        return stats

    def print_summary(self, stats: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        self.logger.info("=== Benchmark Summary ===")
        self.logger.info(f"Model: {stats['model']}")
        self.logger.info(f"Device: {stats['device']}")
        self.logger.info(f"Iterations: {stats['iterations']} (warmup: {stats['warmup_iterations']})")
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Runner")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of benchmark iterations (uses config default if not specified)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run benchmark
    benchmark = BenchmarkRunner(config_path=args.config)
    stats = benchmark.run_benchmark(args.prompt, args.iterations)
    benchmark.print_summary(stats)


if __name__ == "__main__":
    main()
