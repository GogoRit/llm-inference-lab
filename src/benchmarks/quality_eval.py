"""
Quality Evaluation for Speculative Decoding

This module provides quality evaluation tools, including perplexity calculation
for comparing speculative decoding outputs with baseline models.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Evaluates text quality using perplexity scores."""

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize perplexity evaluator.

        Args:
            model_name: Hugging Face model name for evaluation
            device: Device to run evaluation on
        """
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the evaluation model and tokenizer."""
        try:
            self.logger.info(f"Loading evaluation model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map={"": self.device},
                low_cpu_mem_usage=True,
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info(f"Evaluation model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load evaluation model: {e}")
            raise

    def calculate_perplexity(self, text: str) -> Dict[str, Any]:
        """
        Calculate perplexity of the given text.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary containing perplexity metrics
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # Reasonable limit for evaluation
            ).to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = math.exp(loss.item())

            return {
                "perplexity": perplexity,
                "loss": loss.item(),
                "text_length": len(text),
                "token_count": inputs["input_ids"].shape[1],
                "model": self.model_name,
                "device": self.device,
            }

        except Exception as e:
            self.logger.error(f"Perplexity calculation failed: {e}")
            return {
                "perplexity": float("inf"),
                "loss": float("inf"),
                "text_length": len(text),
                "token_count": 0,
                "model": self.model_name,
                "device": self.device,
                "error": str(e),
            }

    def compare_texts(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare perplexity of multiple texts.

        Args:
            texts: List of texts to compare
            labels: Optional labels for the texts

        Returns:
            Dictionary containing comparison results
        """
        if labels is None:
            labels = [f"text_{i}" for i in range(len(texts))]

        results = []
        for text, label in zip(texts, labels):
            result = self.calculate_perplexity(text)
            result["label"] = label
            results.append(result)

        # Calculate statistics
        perplexities = [
            r["perplexity"] for r in results if r["perplexity"] != float("inf")
        ]

        if perplexities:
            avg_perplexity = sum(perplexities) / len(perplexities)
            min_perplexity = min(perplexities)
            max_perplexity = max(perplexities)
        else:
            avg_perplexity = float("inf")
            min_perplexity = float("inf")
            max_perplexity = float("inf")

        return {
            "results": results,
            "statistics": {
                "avg_perplexity": avg_perplexity,
                "min_perplexity": min_perplexity,
                "max_perplexity": max_perplexity,
                "count": len(perplexities),
            },
            "model": self.model_name,
            "device": self.device,
        }

    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def create_evaluator(
    model_name: str = "sshleifer/tiny-gpt2", device: str = "cpu"
) -> PerplexityEvaluator:
    """
    Create a perplexity evaluator with the specified model.

    Args:
        model_name: Hugging Face model name
        device: Device to run on

    Returns:
        PerplexityEvaluator instance
    """
    return PerplexityEvaluator(model_name, device)
