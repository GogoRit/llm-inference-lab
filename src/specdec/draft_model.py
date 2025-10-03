"""
Draft Model for Speculative Decoding

Loads a small draft model (default: distilgpt2) and ensures tokenizer compatibility
with the base model. If different tokenizer families are used, documents the
limitation and runs both tokenizers explicitly.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DraftModel:
    """Small draft model for proposing tokens in speculative decoding."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the draft model.

        Args:
            model_name: Hugging Face model identifier for draft model
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
        """Load the draft model and tokenizer."""
        try:
            self.logger.info(f"Loading draft model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(self.device)

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info(f"Draft model loaded on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load draft model: {e}")
            raise

    def generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        max_draft: int = 4,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens using the draft model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_draft: Maximum number of draft tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (draft_token_ids, draft_logits)
        """
        try:
            with torch.no_grad():
                # Generate draft tokens
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_draft,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs,
                )

                # Extract generated token IDs (excluding input)
                draft_token_ids = outputs.sequences[:, input_ids.shape[1]:]

                # Extract logits for the generated tokens
                if outputs.scores:
                    draft_logits = torch.stack(outputs.scores, dim=1)
                else:
                    # Fallback: get logits from the last layer
                    with torch.no_grad():
                        last_hidden_states = self.model(input_ids).logits
                        draft_logits = last_hidden_states[:, -max_draft:, :]

                return draft_token_ids, draft_logits

        except Exception as e:
            self.logger.error(f"Failed to generate draft tokens: {e}")
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

    def check_tokenizer_compatibility(
        self, base_tokenizer_info: Dict[str, Any]
    ) -> bool:
        """
        Check if this draft model's tokenizer is compatible with the base model.

        Args:
            base_tokenizer_info: Tokenizer info from the base model

        Returns:
            True if compatible, False otherwise
        """
        draft_info = self.get_tokenizer_info()

        # Check if tokenizers are from the same family
        compatible = (
            draft_info["vocab_size"] == base_tokenizer_info["vocab_size"]
            and draft_info["pad_token_id"] == base_tokenizer_info["pad_token_id"]
            and draft_info["eos_token_id"] == base_tokenizer_info["eos_token_id"]
        )

        if not compatible:
            self.logger.warning(
                f"Tokenizer incompatibility detected: "
                f"draft={draft_info}, base={base_tokenizer_info}"
            )
            self.logger.warning(
                "Different tokenizer families detected. This may reduce "
                "acceptance rates. Both tokenizers will be used explicitly for "
                "proposal and verification."
            )

        return compatible
