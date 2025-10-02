"""
Local Baseline Runner for LLM Inference Lab

A minimal Hugging Face Transformers runner that supports both CPU and MPS (Apple Silicon)
for local testing and validation without requiring CUDA.

Usage:
    python -m src.server.local_baseline --prompt "Hello, world!"
    python -m src.server.local_baseline --prompt "The future of AI is" --max-tokens 100
"""

import argparse
import time
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalBaselineRunner:
    """Minimal HF Transformers runner with CPU/MPS support."""

    def __init__(self, model_name: str = "facebook/opt-125m"):
        """
        Initialize the baseline runner.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = self._select_device()
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _select_device(self) -> str:
        """Select the best available device (MPS > CPU)."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_model(self):
        """Load the tokenizer and model."""
        print(f"Loading {self.model_name} on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            device_map="auto" if self.device == "cuda" else None,
        )

        # Move to device if not using device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        print(f"Model loaded successfully on {self.device}")

    def run(self, prompt: str, max_new_tokens: int = 48) -> Dict[str, Any]:
        """
        Run inference on the given prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Dictionary containing device, latency_ms, and generated text
        """
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return {
            "device": self.device,
            "latency_ms": latency_ms,
            "text": generated_text,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Local HF Baseline Runner")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=48,
        help="Maximum number of new tokens to generate (default: 48)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Hugging Face model name (default: facebook/opt-125m)",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = LocalBaselineRunner(model_name=args.model)

    # Run inference
    result = runner.run(args.prompt, args.max_tokens)

    # Print results
    print(f"\n=== Local Baseline Results ===")
    print(f"Device: {result['device']}")
    print(f"Latency: {result['latency_ms']:.2f} ms")
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['text']}")
    print(f"New tokens: {result['max_new_tokens']}")


if __name__ == "__main__":
    main()
