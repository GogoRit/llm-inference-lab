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
        **kwargs,
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

    @property
    def model(self) -> Any:
        """
        Get the underlying model for advanced operations.

        Returns:
            The underlying model object
        """
        return getattr(self, "_model", None)

    @property
    def tokenizer(self) -> Any:
        """
        Get the underlying tokenizer for advanced operations.

        Returns:
            The underlying tokenizer object
        """
        return getattr(self, "_tokenizer", None)

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

    def supports_kv_append(self) -> bool:
        """
        Check if this model supports KV cache appending.

        Returns:
            True if the model can append to KV cache without recomputation
        """
        return False

    def get_kv_cache(self) -> Any:
        """
        Get the current KV cache from the model.

        Returns:
            KVCache object or None if not available
        """
        return None

    def append_kv_cache(self, kv_chunk: Any) -> None:
        """
        Append a KV cache chunk to the model's cache.

        Args:
            kv_chunk: KVCache object containing keys and values to append
        """
        pass

    def clear_kv_cache(self) -> None:
        """Clear the model's KV cache."""
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
        **kwargs,
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
