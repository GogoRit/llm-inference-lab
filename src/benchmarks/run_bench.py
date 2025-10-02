"""
Benchmark Runner for LLM Inference Lab

Runs multiple inference iterations and measures performance metrics including
latency, throughput, and statistical analysis. Supports both local baseline
and HTTP vLLM server endpoints.

Usage:
    # Local baseline benchmarking
    python -m src.benchmarks.run_bench --prompt "Hello, world!" --iterations 5
    
    # HTTP server benchmarking
    python -m src.benchmarks.run_bench --prompt "Hello, world!" --mode http --host 127.0.0.1 --port 8000
    
    # Configuration-driven benchmarking
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
from server.ping_vllm import VLLMPingClient


class BenchmarkRunner:
    """Runs performance benchmarks on local baseline or HTTP vLLM servers."""

    def __init__(self, config_path: str = None, mode: str = "local", 
                 host: str = None, port: int = None):
        """
        Initialize the benchmark runner.

        Args:
            config_path: Path to YAML configuration file
            mode: Benchmark mode ("local" or "http")
            host: Server hostname (for HTTP mode)
            port: Server port (for HTTP mode)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.mode = mode
        
        if mode == "local":
            self.runner = LocalBaselineRunner(
                model_name=self.config.get("model", "facebook/opt-125m"),
                config_path=config_path
            )
            self.runner_name = "LocalBaselineRunner"
        elif mode == "http":
            self.runner = VLLMPingClient(
                host=host or self.config.get("host", "127.0.0.1"),
                port=port or self.config.get("port", 8000),
                config_path=config_path
            )
            self.runner_name = "VLLMPingClient"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'http'")

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
        
        self.logger.info(f"Starting {self.mode} benchmark: {iterations} iterations, {warmup_iterations} warmup")
        self.logger.info(f"Prompt: '{prompt}'")
        
        if self.mode == "local":
            self.logger.info(f"Device: {self.runner.device}")
        else:
            self.logger.info(f"Server: {self.runner.base_url}")
        
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
        results = []
        
        self.logger.info("Starting benchmark iterations...")
        for i in range(iterations):
            if self.mode == "local":
                result = self.runner.run(prompt, self.config.get("max_new_tokens", 48))
                latency_ms = result["latency_ms"]
                tokens_generated = result["max_new_tokens"]
            else:
                result = self.runner.generate(prompt, self.config.get("max_tokens", 50))
                if not result["success"]:
                    self.logger.error(f"Iteration {i+1} failed: {result['error']}")
                    continue
                latency_ms = result["latency_ms"]
                tokens_generated = result["tokens_generated"]
            
            tokens_per_sec = tokens_generated / (latency_ms / 1000.0)
            
            latencies.append(latency_ms)
            tokens_per_second.append(tokens_per_sec)
            results.append(result)
            
            self.logger.debug(f"Iteration {i+1}: {latency_ms:.2f}ms, {tokens_per_sec:.2f} tokens/sec")
        
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
        
        # Add mode-specific information
        if self.mode == "local":
            stats["device"] = self.runner.device
            stats["model"] = self.runner.model_name
        else:
            stats["server"] = self.runner.base_url
            stats["model"] = self.config.get("model", "unknown")
        
        return stats

    def print_summary(self, stats: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        self.logger.info("=== Benchmark Summary ===")
        self.logger.info(f"Mode: {stats['mode']} ({stats['runner_name']})")
        self.logger.info(f"Model: {stats['model']}")
        
        if stats['mode'] == "local":
            self.logger.info(f"Device: {stats['device']}")
        else:
            self.logger.info(f"Server: {stats['server']}")
            
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
        "--mode",
        type=str,
        choices=["local", "http"],
        default="local",
        help="Benchmark mode: local baseline or HTTP server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server hostname (for HTTP mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (for HTTP mode)"
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
    benchmark = BenchmarkRunner(
        config_path=args.config,
        mode=args.mode,
        host=args.host,
        port=args.port
    )
    stats = benchmark.run_benchmark(args.prompt, args.iterations)
    
    if "error" in stats:
        benchmark.logger.error(f"Benchmark failed: {stats['error']}")
    else:
        benchmark.print_summary(stats)


if __name__ == "__main__":
    main()
