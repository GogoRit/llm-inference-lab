"""
Local Baseline Runner for LLM Inference Lab

A minimal Hugging Face Transformers runner that supports both CPU and MPS
(Apple Silicon) for local testing and validation without requiring CUDA.

Usage:
    python -m src.server.local_baseline --prompt "Hello, world!"
    python -m src.server.local_baseline --prompt "The future of AI is" --max-tokens 100
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalBaselineRunner:
    """Minimal HF Transformers runner with CPU/MPS support."""

    def __init__(
        self, model_name: str = "facebook/opt-125m", config_path: Optional[str] = None
    ):
        """
        Initialize the baseline runner.

        Args:
            model_name: Hugging Face model identifier
            config_path: Path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.model_name = model_name
        self.device = self._select_device()
        self.tokenizer: Any = None
        self.model: Any = None
        self._load_model()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "model": "facebook/opt-125m",
            "max_new_tokens": 48,
            "temperature": 0.7,
            "do_sample": True,
            "torch_dtype": "float32",
            "device_priority": ["mps", "cuda", "cpu"],
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                self.logger.info("Using default configuration")
        else:
            self.logger.info("Using default configuration")

        return default_config

    def _select_device(self) -> str:
        """Select the best available device based on config priority."""
        device_priority = self.config.get("device_priority", ["mps", "cuda", "cpu"])

        for device in device_priority:
            if device == "mps" and torch.backends.mps.is_available():
                self.logger.info("Selected MPS device (Apple Silicon GPU)")
                return "mps"
            elif device == "cuda" and torch.cuda.is_available():
                self.logger.info("Selected CUDA device")
                return "cuda"
            elif device == "cpu":
                self.logger.info("Selected CPU device")
                return "cpu"

        # Fallback to CPU if nothing else works
        self.logger.warning("No preferred device available, falling back to CPU")
        return "cpu"

    def _load_model(self):
        """Load the tokenizer and model."""
        self.logger.info(f"Loading {self.model_name} on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine torch dtype from config
        torch_dtype_str = self.config.get("torch_dtype", "float32")
        torch_dtype = getattr(torch, torch_dtype_str)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Move to device if not using device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.logger.info(f"Model loaded successfully on {self.device}")

    def run(self, prompt: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Run inference on the given prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
                (uses config if None)

        Returns:
            Dictionary containing device, latency_ms, and generated text
        """
        start_time = time.time()

        # Use config default if max_new_tokens not provided
        if max_new_tokens is None:
            max_new_tokens = self.config.get("max_new_tokens", 48)

        # Ensure model and tokenizer are loaded
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert self.model is not None, "Model not loaded"

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate using config parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.get("do_sample", True),
                temperature=self.config.get("temperature", 0.7),
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
        default=None,
        help="Maximum number of new tokens to generate "
        "(uses config default if not specified)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Hugging Face model name (default: facebook/opt-125m)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize runner
    runner = LocalBaselineRunner(model_name=args.model, config_path=args.config)

    # Run inference
    result = runner.run(args.prompt, args.max_tokens)

    # Log results
    runner.logger.info("=== Local Baseline Results ===")
    runner.logger.info(f"Device: {result['device']}")
    runner.logger.info(f"Latency: {result['latency_ms']:.2f} ms")
    runner.logger.info(f"Prompt: {result['prompt']}")
    runner.logger.info(f"Generated: {result['text']}")
    runner.logger.info(f"New tokens: {result['max_new_tokens']}")


if __name__ == "__main__":
    main()
