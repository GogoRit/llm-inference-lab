#!/usr/bin/env python3
"""
CUDA Smoke Test for Speculative Decoding

Tests CUDA functionality with mixed precision, device detection,
and basic speculative decoding pipeline.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch

from specdec.pipeline import SpeculativePipeline

# Set environment variables for testing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("SPECULATIVE DECODING CUDA SMOKE TEST")
    print("=" * 60)

    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        expected_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    elif torch.backends.mps.is_available():
        device = "mps"
        expected_dtype = "float16"
    else:
        device = "cpu"
        expected_dtype = "float32"

    print(f"Using device: {device.upper()}")
    print(f"Expected dtype: {expected_dtype}")
    print("-" * 60)

    try:
        # Test with AMP enabled
        print("Testing with AMP optimization...")
        pipe = SpeculativePipeline(
            base_model="gpt2",
            draft_model="distilgpt2",
            max_draft=2,
            implementation="hf",
            enable_optimization=True,
            device=device,
            seed=42,  # For reproducibility
        )

        # Test generation
        res = pipe.generate(
            "Say hi in 1 sentence.", max_tokens=16, temperature=0.7, do_sample=True
        )

        print(f"Generated text: {res['text']}")
        print(f"Device: {res.get('device', 'unknown')}")
        print(f"Dtype: {res.get('dtype', 'unknown')}")
        print(f"AMP enabled: {res.get('amp_enabled', 'unknown')}")
        print(f"Acceptance rate: {res.get('acceptance_rate', 0.0):.3f}")
        print(f"Tokens/sec: {res.get('tokens_per_sec', 0.0):.2f}")

        if device == "cuda":
            print(
                f"CUDA memory allocated: {res.get('cuda_mem_allocated_mb', 0):.1f} MB"
            )
            print(f"CUDA memory peak: {res.get('cuda_mem_peak_mb', 0):.1f} MB")

        # Validation
        assert res["text"] is not None, "Generated text is None"
        assert len(res.get("generated_tokens", [])) <= 16, "Generated too many tokens"
        assert (
            res.get("device") == device
        ), f"Device mismatch: expected {device}, got {res.get('device')}"

        # Check dtype (may vary based on device capabilities)
        actual_dtype = res.get("dtype", "unknown")
        if device in ["cuda", "mps"]:
            assert actual_dtype in [
                "float16",
                "bfloat16",
                "torch.float16",
                "torch.bfloat16",
            ], f"Unexpected dtype: {actual_dtype}"
        else:
            assert actual_dtype in [
                "float32",
                "torch.float32",
            ], f"Expected float32 on CPU, got {actual_dtype}"

        print("-" * 60)
        print("All smoke tests passed!")
        print(f"Device: {device.upper()}")
        print(f"Dtype: {actual_dtype}")
        print(f"AMP: {res.get('amp_enabled', False)}")
        print(f"Generated {len(res.get('generated_tokens', []))} tokens")
        print("=" * 60)

    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
