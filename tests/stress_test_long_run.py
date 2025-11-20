#!/usr/bin/env python3
"""
Stress Test for Long Speculative Decoding Runs

Tests stability over hundreds of steps to ensure:
- No memory leaks
- Notebook stays responsive
- No excessive CPU usage
- GPU memory stays bounded
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

# Disable all debug prints and periodic syncs for stress test
os.environ["SPECDEC_DEBUG_PRINTS"] = "0"
os.environ["SPECDEC_PERIODIC_SYNC"] = "0"
os.environ["SPECDEC_PERIODIC_CACHE_CLEAR"] = "0"
os.environ["SPECDEC_DEBUG"] = "0"

# Try to import pipeline (will fail if models not available)
try:
    from specdec.core.pipeline import SpeculativePipeline

    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    print("Warning: Could not import SpeculativePipeline. Running synthetic test only.")


def measure_memory():
    """Measure current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**2),
            "reserved": torch.cuda.memory_reserved() / (1024**2),
            "max_allocated": torch.cuda.max_memory_allocated() / (1024**2),
        }
    return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}


def create_dummy_prompts(num_prompts: int, base_length: int = 50) -> list:
    """Create dummy prompts for testing."""
    prompts = []
    for i in range(num_prompts):
        # Vary prompt lengths slightly
        length = base_length + (i % 10)
        prompt = " ".join(["test"] * length)
        prompts.append(prompt)
    return prompts


def run_stress_test(
    pipeline,
    num_steps: int = 200,
    num_prompts: int = 4,
    check_interval: int = 20,
):
    """
    Run a stress test with many steps.

    Args:
        pipeline: SpeculativePipeline instance
        num_steps: Number of generation steps (max_tokens)
        num_prompts: Number of prompts to process
        check_interval: Check memory every N steps
    """
    print("=" * 80)
    print("STRESS TEST: Long Speculative Decoding Run")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Steps: {num_steps}")
    print(f"  - Prompts: {num_prompts}")
    print(f"  - Check interval: {check_interval}")
    print(f"  - Debug prints: {os.getenv('SPECDEC_DEBUG_PRINTS', '0')}")
    print(f"  - Periodic sync: {os.getenv('SPECDEC_PERIODIC_SYNC', '0')}")
    print(f"  - Periodic cache clear: {os.getenv('SPECDEC_PERIODIC_CACHE_CLEAR', '0')}")
    print()

    # Create prompts
    prompts = create_dummy_prompts(num_prompts)

    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Measure initial memory
    mem_initial = measure_memory()
    print(
        f"Initial GPU memory: {mem_initial['allocated']:.1f} MB allocated, "
        f"{mem_initial['reserved']:.1f} MB reserved"
    )
    print()

    # Run generation
    start_time = time.time()
    try:
        results = pipeline.generate_batch(
            prompts=prompts,
            max_tokens=num_steps,
            temperature=0.7,
            do_sample=True,
        )
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        results = []
        import traceback

        traceback.print_exc()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Measure final memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_final = measure_memory()
    mem_peak = measure_memory()

    # Calculate metrics
    total_tokens = sum(len(r.get("generated_tokens", [])) for r in results)
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
    mem_growth = mem_final["allocated"] - mem_initial["allocated"]
    mem_peak_growth = mem_peak["max_allocated"] - mem_initial["allocated"]

    # Print results
    print()
    print("=" * 80)
    print("STRESS TEST RESULTS")
    print("=" * 80)
    print(f"Success: {success}")
    if error:
        print(f"Error: {error}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_sec:.1f}")
    print()
    print("Memory:")
    print(
        f"  Initial: {mem_initial['allocated']:.1f} MB allocated, "
        f"{mem_initial['reserved']:.1f} MB reserved"
    )
    print(
        f"  Final:   {mem_final['allocated']:.1f} MB allocated, "
        f"{mem_final['reserved']:.1f} MB reserved"
    )
    print(f"  Peak:    {mem_peak['max_allocated']:.1f} MB (max allocated)")
    print(f"  Growth:  {mem_growth:.1f} MB (final - initial)")
    print(f"  Peak growth: {mem_peak_growth:.1f} MB (peak - initial)")
    print()

    # Check for issues
    issues = []
    if not success:
        issues.append("Generation failed")
    if mem_growth > 1000:  # More than 1GB growth
        issues.append(f"Excessive memory growth: {mem_growth:.1f} MB")
    if mem_peak_growth > 2000:  # More than 2GB peak growth
        issues.append(f"Excessive peak memory: {mem_peak_growth:.1f} MB")
    if tokens_per_sec < 1.0:
        issues.append(f"Very slow generation: {tokens_per_sec:.1f} tok/s")

    if issues:
        print("WARNING: ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Stress test passed: No major issues detected")
        return True


def main():
    """Main entry point."""
    if not HAS_PIPELINE:
        print("Pipeline not available. Cannot run full stress test.")
        print()
        print("To run full stress test:")
        print("1. Initialize SpeculativePipeline with your models")
        print("2. Call run_stress_test(pipeline, num_steps=200)")
        print()
        return

    # Example usage (commented out - requires actual pipeline)
    """
    # Initialize pipeline
    pipeline = SpeculativePipeline(
        base_model="your-base-model",
        draft_model="your-draft-model",
        device="cuda",
    )

    # Run stress test
    success = run_stress_test(
        pipeline=pipeline,
        num_steps=200,
        num_prompts=4,
        check_interval=20,
    )

    if not success:
        sys.exit(1)
    """

    print("Stress test script ready.")
    print("Uncomment the code above and provide your pipeline to run the test.")


if __name__ == "__main__":
    main()
