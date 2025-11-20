#!/usr/bin/env python3
"""
Microbenchmark for Sequence Pool vs Traditional Padding

Compares performance of sequence pool (EXSPEC-style) vs traditional
padding-to-max approach on synthetic workloads.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

# Mock models for benchmarking (we'll use actual models if available)
try:
    from specdec.core.pipeline import SpeculativePipeline
    from specdec.models.hf_wrappers import HuggingFaceLanguageModel

    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False
    print("Warning: Could not import models. Running synthetic benchmark only.")


def generate_synthetic_prompts(
    num_prompts: int,
    length_distribution: str = "normal",
    mean_length: int = 50,
    std_length: int = 10,
    min_length: int = 10,
    max_length: int = 200,
) -> list:
    """
    Generate synthetic prompts with specified length distribution.

    Args:
        num_prompts: Number of prompts to generate
        length_distribution: "normal" or "skewed"
        mean_length: Mean length for normal distribution
        std_length: Standard deviation for normal distribution
        min_length: Minimum prompt length
        max_length: Maximum prompt length

    Returns:
        List of prompt strings (just dummy text, length matters)
    """
    prompts = []

    if length_distribution == "normal":
        lengths = np.random.normal(mean_length, std_length, num_prompts)
    elif length_distribution == "skewed":
        # Highly skewed: most sequences short, few very long
        lengths = np.random.exponential(mean_length / 2, num_prompts)
    else:
        raise ValueError(f"Unknown distribution: {length_distribution}")

    # Clamp to valid range
    lengths = np.clip(lengths, min_length, max_length).astype(int)

    # Generate prompts (just dummy text, actual content doesn't matter)
    for length in lengths:
        prompt = " ".join(["word"] * length)
        prompts.append(prompt)

    return prompts, lengths.tolist()


def measure_memory_usage():
    """Measure current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def run_benchmark(
    pipeline,
    prompts: list,
    max_tokens: int = 20,
    use_sequence_pool: bool = False,
) -> dict:
    """
    Run a single benchmark iteration.

    Returns:
        Dictionary with metrics
    """
    # Set environment variable
    if use_sequence_pool:
        os.environ["SPECDEC_ENABLE_SEQUENCE_POOL"] = "1"
    else:
        os.environ["SPECDEC_ENABLE_SEQUENCE_POOL"] = "0"

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Measure memory before
    mem_before = measure_memory_usage()

    # Run generation
    start_time = time.time()
    try:
        results = pipeline.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
        )
        success = True
    except Exception as e:
        print(f"Error during generation: {e}")
        success = False
        results = []

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Measure memory after
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_after = measure_memory_usage()
    mem_used = mem_after - mem_before

    # Calculate metrics
    total_tokens = sum(len(r.get("generated_tokens", [])) for r in results)
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0.0

    return {
        "success": success,
        "elapsed_time": elapsed_time,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "memory_used_mb": mem_used,
        "num_prompts": len(prompts),
    }


def main():
    """Run microbenchmark comparing sequence pool vs traditional padding."""
    print("=" * 80)
    print("SEQUENCE POOL MICROBENCHMARK")
    print("=" * 80)
    print()

    if not HAS_MODELS:
        print("Models not available. Exiting.")
        return

    # Configuration
    num_prompts = 8
    max_tokens = 20
    num_runs = 3

    # Test configurations
    test_configs = [
        {
            "name": "Normal Distribution",
            "distribution": "normal",
            "mean_length": 50,
            "std_length": 10,
        },
        {
            "name": "Skewed Distribution",
            "distribution": "skewed",
            "mean_length": 50,
            "std_length": 10,
        },
    ]

    # Initialize pipeline (you'll need to provide actual models)
    print("Note: This benchmark requires actual model initialization.")
    print("Please modify this script to use your models.")
    print()

    # For now, just demonstrate the synthetic prompt generation
    print("Generating synthetic prompts...")
    for config in test_configs:
        print(f"\n{config['name']}:")
        prompts, lengths = generate_synthetic_prompts(
            num_prompts=num_prompts,
            length_distribution=config["distribution"],
            mean_length=config["mean_length"],
            std_length=config["std_length"],
        )

        print(f"  Generated {len(prompts)} prompts")
        print(
            f"  Length stats: min={min(lengths)}, max={max(lengths)}, "
            f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}"
        )
        print(f"  Length distribution: {lengths}")

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("To run full benchmark:")
    print("1. Initialize SpeculativePipeline with your models")
    print("2. Uncomment the benchmark runs below")
    print("3. Run: python benchmarks/benchmark_sequence_pool.py")
    print()

    # Example benchmark structure (commented out)
    """
    pipeline = SpeculativePipeline(...)  # Initialize with your models
    
    for config in test_configs:
        print(f"\n{config['name']}:")
        prompts, lengths = generate_synthetic_prompts(
            num_prompts=num_prompts,
            length_distribution=config["distribution"],
            mean_length=config["mean_length"],
            std_length=config["std_length"],
        )
        
        # Run without sequence pool
        print("  Running without sequence pool...")
        results_no_pool = []
        for _ in range(num_runs):
            result = run_benchmark(pipeline, prompts, max_tokens, use_sequence_pool=False)
            results_no_pool.append(result)
        
        # Run with sequence pool
        print("  Running with sequence pool...")
        results_with_pool = []
        for _ in range(num_runs):
            result = run_benchmark(pipeline, prompts, max_tokens, use_sequence_pool=True)
            results_with_pool.append(result)
        
        # Calculate averages
        avg_no_pool = {
            "tokens_per_sec": np.mean([r["tokens_per_sec"] for r in results_no_pool]),
            "memory_mb": np.mean([r["memory_used_mb"] for r in results_no_pool]),
        }
        avg_with_pool = {
            "tokens_per_sec": np.mean([r["tokens_per_sec"] for r in results_with_pool]),
            "memory_mb": np.mean([r["memory_used_mb"] for r in results_with_pool]),
        }
        
        # Print results
        print(f"  Without pool: {avg_no_pool['tokens_per_sec']:.1f} tok/s, "
              f"{avg_no_pool['memory_mb']:.1f} MB")
        print(f"  With pool:    {avg_with_pool['tokens_per_sec']:.1f} tok/s, "
              f"{avg_with_pool['memory_mb']:.1f} MB")
        print(f"  Speedup:      {avg_with_pool['tokens_per_sec'] / avg_no_pool['tokens_per_sec']:.2f}x")
        print(f"  Memory saved: {avg_no_pool['memory_mb'] - avg_with_pool['memory_mb']:.1f} MB")
    """


if __name__ == "__main__":
    main()
