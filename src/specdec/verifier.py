"""
Verifier for Speculative Decoding

Given (prompt + proposed tokens), runs the base model to verify and returns the
length of the accepted prefix. Uses exact-match policy: accept consecutive tokens
while they match the base model's greedy continuation.
"""

import logging
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Verifier:
    """Base model verifier for speculative decoding."""

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the verifier.

        Args:
            model_name: Hugging Face model identifier for base model
            device: Device to run on ("auto", "cpu", "mps", "cuda")
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = self._select_device(device)
        self.config = config or {}

        self.tokenizer: Any = None
        self.model: Any = None
        self._load_model()

    def _select_device(self, device: str) -> str:
        """Select the best available device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the base model and tokenizer."""
        try:
            self.logger.info(f"Loading base model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(self.device)

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info(f"Base model loaded on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise

    def verify_tokens(
        self,
        input_ids: torch.Tensor,
        proposed_tokens: torch.Tensor,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """
        Verify proposed tokens against the base model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            proposed_tokens: Proposed token IDs [batch_size, num_proposed]
            temperature: Sampling temperature (for fallback generation)
            do_sample: Whether to use sampling (for fallback generation)

        Returns:
            Dictionary containing:
            - accepted_len: Number of accepted tokens
            - accepted_tokens: Accepted token IDs
            - verification_time_ms: Time taken for verification
            - base_tokens: Base model's greedy continuation
        """
        start_time = time.time()

        try:
            with torch.no_grad():
                # Get base model's greedy continuation for the same input
                base_outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=proposed_tokens.shape[1],
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )

                # Extract base model's generated tokens (excluding input)
                base_tokens = base_outputs.sequences[:, input_ids.shape[1] :]

                # Find the longest matching prefix
                accepted_len = self._find_accepted_length(proposed_tokens, base_tokens)
                accepted_tokens = (
                    proposed_tokens[:, :accepted_len]
                    if accepted_len > 0
                    else torch.empty(
                        proposed_tokens.shape[0], 0, dtype=proposed_tokens.dtype
                    )
                )

                verification_time_ms = (time.time() - start_time) * 1000

                return {
                    "accepted_len": accepted_len,
                    "accepted_tokens": accepted_tokens,
                    "verification_time_ms": verification_time_ms,
                    "base_tokens": base_tokens,
                    "proposed_tokens": proposed_tokens,
                }

        except Exception as e:
            self.logger.error(f"Failed to verify tokens: {e}")
            raise

    def _find_accepted_length(
        self, proposed_tokens: torch.Tensor, base_tokens: torch.Tensor
    ) -> int:
        """
        Find the length of the longest matching prefix between proposed and base tokens.

        Args:
            proposed_tokens: Proposed token IDs [batch_size, num_proposed]
            base_tokens: Base model token IDs [batch_size, num_base]

        Returns:
            Length of accepted prefix
        """
        # For simplicity, we'll work with the first batch item
        proposed = proposed_tokens[0].cpu().numpy()
        base = base_tokens[0].cpu().numpy()

        # Find the minimum length to compare
        min_len = min(len(proposed), len(base))

        # Count consecutive matching tokens
        accepted_len = 0
        for i in range(min_len):
            if proposed[i] == base[i]:
                accepted_len += 1
            else:
                break

        return accepted_len

    def generate_fallback_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a single fallback token when no proposed tokens are accepted.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Dictionary containing the fallback token and generation time
        """
        start_time = time.time()

        try:
            with torch.no_grad():
                # Generate exactly one token
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )

                # Extract the generated token (excluding input)
                fallback_token = outputs.sequences[:, input_ids.shape[1] :]
                generation_time_ms = (time.time() - start_time) * 1000

                return {
                    "fallback_token": fallback_token,
                    "generation_time_ms": generation_time_ms,
                }

        except Exception as e:
            self.logger.error(f"Failed to generate fallback token: {e}")
            raise

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer information for compatibility checking."""
        return {
            "model_name": self.model_name,
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
        }
