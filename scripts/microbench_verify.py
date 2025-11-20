#!/usr/bin/env python3
"""
Microbenchmark for verify_prefix kernel performance.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernels import get_kernel_info, verify_prefix
from kernels.reference import verify_prefix_ref


def benchmark_verify_prefix():
    """Benchmark verify_prefix kernel vs PyTorch reference."""
    print("=" * 80)
    print("VERIFY_PREFIX KERNEL MICROBENCHMARK")
    print("=" * 80)

    # Get kernel info
    kernel_info = get_kernel_info()
    print(f"Kernel backend: {kernel_info['verify_backend']}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Test configurations
    test_configs = [
        # (B, K, V, description)
        (1, 1, 4096, "Single token, small vocab"),
        (1, 2, 4096, "K=2, small vocab"),
        (1, 4, 4096, "K=4, small vocab"),
        (8, 1, 4096, "Batch=8, single token"),
        (8, 2, 4096, "Batch=8, K=2"),
        (8, 4, 4096, "Batch=8, K=4"),
        (1, 1, 8192, "Single token, medium vocab"),
        (1, 2, 8192, "K=2, medium vocab"),
        (1, 4, 8192, "K=4, medium vocab"),
        (1, 1, 32768, "Single token, large vocab"),
        (1, 2, 32768, "K=2, large vocab"),
        (1, 4, 32768, "K=4, large vocab"),
    ]

    results = []

    for B, K, V, description in test_configs:
        print(f"Testing: {description} (B={B}, K={K}, V={V})")

        # Generate test data
        logits = torch.randn(B, K, V)
        draft_ids = torch.randint(0, V, (B, K))

        # Set up some matches for realistic testing
        for b in range(B):
            for k in range(K):
                if np.random.random() < 0.3:  # 30% chance of match
                    logits[b, k, draft_ids[b, k]] = 10.0

        # Move to CUDA if available
        if torch.cuda.is_available():
            logits = logits.cuda()
            draft_ids = draft_ids.cuda()

        # Warmup
        for _ in range(10):
            if kernel_info["verify_backend"] != "fallback":
                try:
                    verify_prefix(logits, draft_ids)
                except Exception:
                    pass
            verify_prefix_ref(logits, draft_ids)

        # Benchmark kernel (if available)
        kernel_times = []
        if kernel_info["verify_backend"] != "fallback":
            try:
                for _ in range(100):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start = time.time()
                    verify_prefix(logits, draft_ids)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    kernel_times.append(time.time() - start)
            except Exception as e:
                print(f"  Kernel failed: {e}")
                kernel_times = []

        # Benchmark reference
        ref_times = []
        for _ in range(100):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            verify_prefix_ref(logits, draft_ids)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            ref_times.append(time.time() - start)

        # Calculate statistics
        if kernel_times:
            kernel_mean = np.mean(kernel_times) * 1000  # Convert to ms
            kernel_std = np.std(kernel_times) * 1000
        else:
            kernel_mean = float("inf")
            kernel_std = 0

        ref_mean = np.mean(ref_times) * 1000
        ref_std = np.std(ref_times) * 1000

        speedup = ref_mean / kernel_mean if kernel_mean != float("inf") else 0

        # Store results
        results.append(
            {
                "config": f"B={B}, K={K}, V={V}",
                "description": description,
                "kernel_time_ms": kernel_mean,
                "kernel_std_ms": kernel_std,
                "ref_time_ms": ref_mean,
                "ref_std_ms": ref_std,
                "speedup": speedup,
            }
        )

        # Print results
        if kernel_times:
            print(f"  Kernel: {kernel_mean:.3f}±{kernel_std:.3f} ms")
            print(f"  Reference: {ref_mean:.3f}±{ref_std:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print(f"  Reference: {ref_mean:.3f}±{ref_std:.3f} ms")
            print(f"  Kernel: Not available")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Kernel (ms)':<12} {'Reference (ms)':<15} {'Speedup':<8}")
    print("-" * 80)

    for result in results:
        if result["kernel_time_ms"] != float("inf"):
            print(
                f"{result['config']:<20} {result['kernel_time_ms']:.3f}±{result['kernel_std_ms']:.1f}    "
                f"{result['ref_time_ms']:.3f}±{result['ref_std_ms']:.1f}      {result['speedup']:.2f}x"
            )
        else:
            print(
                f"{result['config']:<20} {'N/A':<12} {result['ref_time_ms']:.3f}±{result['ref_std_ms']:.1f}      N/A"
            )

    # Calculate average speedup
    valid_speedups = [r["speedup"] for r in results if r["speedup"] > 0]
    if valid_speedups:
        avg_speedup = np.mean(valid_speedups)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")

        if avg_speedup >= 5.0:
            print("Target speedup (5x) achieved!")
        else:
            print(
                f"WARNING: Target speedup (5x) not achieved. Current: {avg_speedup:.2f}x"
            )
    else:
        print("\nWARNING: No valid speedup measurements available")


if __name__ == "__main__":
    benchmark_verify_prefix()
