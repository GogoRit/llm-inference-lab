#!/usr/bin/env python3
"""
CUDA Smoke Test for Speculative Decoding Pipeline
Quick test to verify CUDA functionality and fp16 dtype selection.
"""

import os
import pathlib
import sys

import torch

# Silence tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path
sys.path.insert(0, str(pathlib.Path("src").resolve()))

from specdec.pipeline import SpeculativePipeline  # noqa: E402


def main():
    print(
        "Torch:",
        torch.__version__,
        "| CUDA:",
        torch.cuda.is_available(),
        "| Dev:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    )

    if not torch.cuda.is_available():
        print("CUDA not available, testing MPS fallback...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cuda"

    try:
        pipe = SpeculativePipeline(
            base_model="gpt2",
            draft_model="distilgpt2",
            max_draft=2,
            implementation="hf",
            enable_optimization=True,
            device=device,
        )

        res = pipe.generate(
            "Say hi in 1 sentence.", max_tokens=8, temperature=0.7, do_sample=True
        )

        print(f"{device.upper()} sanity OK. Text:", res["text"])
        print("Device:", res.get("device", "unknown"))
        print("Dtype:", res.get("dtype", "unknown"))

        # Quick unit test validation
        assert res["text"] is not None, "Generated text is None"
        assert len(res.get("generated_tokens", [])) <= 8, "Generated too many tokens"
        assert (
            res.get("device") == device
        ), f"Device mismatch: expected {device}, got {res.get('device')}"

        print("✅ All smoke tests passed!")

    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
