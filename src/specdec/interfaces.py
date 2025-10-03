"""
Language Model Interface for Speculative Decoding

Defines a common interface for language models to enable dependency injection
and support both real Hugging Face models and fake models for testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class LanguageModel(ABC):
    """Abstract base class for language models in speculative decoding."""

    @abstractmethod
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens from the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_token_ids, logits)
        """
        pass

    @abstractmethod
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        Get tokenizer information for compatibility checking.

        Returns:
            Dictionary containing tokenizer metadata
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            Token IDs tensor
        """
        pass

    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device this model is running on."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        pass


class SpeculativeDecoder(ABC):
    """Abstract base class for speculative decoding algorithms."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generation results and metrics
        """
        pass
