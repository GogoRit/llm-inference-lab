"""
Fake Language Model for Testing

Provides FakeLM that generates deterministic tokens for unit tests
without loading actual models. Useful for testing speculative decoding
logic without memory issues.
"""

import logging
import random
from typing import Any, Dict, Optional, Tuple

import torch

from .interfaces import LanguageModel


class FakeLM(LanguageModel):
    """Fake language model that generates deterministic tokens for testing."""

    def __init__(
        self,
        model_name: str = "fake-model",
        vocab_size: int = 1000,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize the fake language model.

        Args:
            model_name: Fake model name for identification
            vocab_size: Vocabulary size for token generation
            device: Device to simulate (always CPU for fake model)
            seed: Random seed for deterministic generation
        """
        self.logger = logging.getLogger(__name__)
        self._model_name = model_name
        self._vocab_size = vocab_size
        self._device = device
        self._seed = seed

        # Set up deterministic generation if seed provided
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Tokenizer info
        self._pad_token_id = 0
        self._eos_token_id = 1
        self._bos_token_id = 2
        self._unk_token_id = 3

        self.logger.info(f"FakeLM initialized: {model_name} (vocab_size={vocab_size})")

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate fake tokens deterministically.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (ignored for deterministic generation)
            do_sample: Whether to use sampling (ignored for deterministic generation)
            **kwargs: Additional generation parameters (ignored)

        Returns:
            Tuple of (generated_token_ids, fake_logits)
        """
        # Generate deterministic tokens based on input
        # Use a simple hash of the input to create deterministic output
        input_hash = hash(tuple(input_ids[0].tolist()))

        # Generate tokens deterministically
        generated_tokens = []
        for i in range(max_new_tokens):
            # Create deterministic token based on input hash and position
            token_id = (input_hash + i) % self._vocab_size
            # Avoid special tokens
            if token_id in [
                self._pad_token_id,
                self._eos_token_id,
                self._bos_token_id,
                self._unk_token_id,
            ]:
                token_id = (token_id + 1) % self._vocab_size
            generated_tokens.append(token_id)

        # Convert to tensor on the same device as input
        generated_ids = torch.tensor(
            [generated_tokens], dtype=torch.long, device=input_ids.device
        )

        # Generate fake logits (uniform distribution) on the same device
        logits = torch.randn(
            1, max_new_tokens, self._vocab_size, device=input_ids.device
        )

        return generated_ids, logits

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get fake tokenizer information."""
        return {
            "model_name": self._model_name,
            "vocab_size": self._vocab_size,
            "pad_token_id": self._pad_token_id,
            "eos_token_id": self._eos_token_id,
            "bos_token_id": self._bos_token_id,
            "unk_token_id": self._unk_token_id,
        }

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to fake token IDs."""
        # Simple deterministic encoding based on text hash
        text_hash = hash(text)
        # Generate 3-5 tokens based on text length
        num_tokens = min(max(3, len(text) // 2), 5)
        tokens = [(text_hash + i) % self._vocab_size for i in range(num_tokens)]
        return torch.tensor([tokens], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode fake token IDs to text."""
        # Simple deterministic decoding
        if token_ids.numel() == 0:
            return ""

        # Create fake text based on token IDs
        token_list = token_ids[0].tolist()
        fake_text = f"fake_text_{'_'.join(map(str, token_list[:3]))}"
        return fake_text

    @property
    def device(self) -> str:
        """Get the device this model is running on."""
        return self._device

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name


class FakeLMWithAcceptance(FakeLM):
    """Fake language model that simulates acceptance patterns for testing."""

    def __init__(
        self,
        model_name: str = "fake-model-with-acceptance",
        vocab_size: int = 1000,
        device: str = "cpu",
        seed: Optional[int] = None,
        acceptance_rate: float = 0.7,
    ):
        """
        Initialize fake LM with configurable acceptance rate.

        Args:
            model_name: Fake model name for identification
            vocab_size: Vocabulary size for token generation
            device: Device to simulate
            seed: Random seed for deterministic generation
            acceptance_rate: Probability of accepting proposed tokens
        """
        super().__init__(model_name, vocab_size, device, seed)
        self._acceptance_rate = acceptance_rate
        self._call_count = 0
        if seed is not None:
            random.seed(seed)
        self.logger.info(
            f"FakeLMWithAcceptance initialized with acceptance_rate={acceptance_rate}"
        )

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens with configurable acceptance rate."""
        # Generate base tokens
        generated_ids, logits = super().generate_tokens(
            input_ids, max_new_tokens, temperature, do_sample, **kwargs
        )

        # Simulate acceptance by potentially truncating the sequence
        # Use deterministic random based on input for reproducibility
        input_hash = hash(tuple(input_ids[0].tolist()))
        # Add call count to create variation between calls (for testing acceptance rate)
        self._call_count += 1
        random_seed = (
            input_hash
            + (self._seed if self._seed is not None else 0)
            + self._call_count
        )
        random.seed(random_seed)
        if random.random() < self._acceptance_rate:
            # Accept some tokens (truncate randomly)
            accept_count = random.randint(1, max_new_tokens)
            generated_ids = generated_ids[:, :accept_count]
            logits = logits[:, :accept_count, :]

        return generated_ids, logits


def create_fake_lm(
    model_name: str = "fake-model",
    vocab_size: int = 1000,
    device: str = "cpu",
    seed: Optional[int] = None,
    acceptance_rate: Optional[float] = None,
) -> LanguageModel:
    """
    Create a fake language model for testing.

    Args:
        model_name: Fake model name
        vocab_size: Vocabulary size
        device: Device to simulate
        seed: Random seed for deterministic generation
        acceptance_rate: If provided, creates FakeLMWithAcceptance

    Returns:
        FakeLM or FakeLMWithAcceptance instance
    """
    if acceptance_rate is not None:
        return FakeLMWithAcceptance(
            model_name=model_name,
            vocab_size=vocab_size,
            device=device,
            seed=seed,
            acceptance_rate=acceptance_rate,
        )
    else:
        return FakeLM(
            model_name=model_name,
            vocab_size=vocab_size,
            device=device,
            seed=seed,
        )
