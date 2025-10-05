"""
Hugging Face Model Wrappers for Speculative Decoding

Provides HFWrapper that wraps Hugging Face models with safe defaults
for tiny models only (smoke runs) and MPS memory hygiene.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interfaces import LanguageModel


class HFWrapper(LanguageModel):
    """Wrapper for Hugging Face models with memory-safe defaults."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        max_memory_mb: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize the Hugging Face wrapper.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ("auto", "cpu", "mps", "cuda")
            torch_dtype: PyTorch data type for the model (auto-selected if None)
            max_memory_mb: Maximum memory usage in MB (for safety)
            tokenizer: Shared tokenizer instance (to avoid duplicate RAM)
        """
        self.logger = logging.getLogger(__name__)
        self._model_name = model_name
        self._device = self._select_device(device)
        self._torch_dtype = torch_dtype or self._select_dtype()
        self._max_memory_mb = max_memory_mb
        self._tokenizer = tokenizer

        # Load model and tokenizer
        self._load_model()

    def _select_device(self, device: str) -> str:
        """Select the best available device with memory considerations."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _select_dtype(self) -> torch.dtype:
        """Select appropriate dtype based on device."""
        if self._device in ["cuda", "mps"]:
            return torch.float16
        else:
            return torch.float32

    def _load_model(self) -> None:
        """Load the model and tokenizer with memory safety."""
        try:
            self.logger.info(f"Loading HF model: {self._model_name}")

            # Load tokenizer (shared if provided)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                if self._tokenizer.pad_token is None:  # type: ignore
                    self._tokenizer.pad_token = (  # type: ignore
                        self._tokenizer.eos_token  # type: ignore
                    )
                self._tokenizer.padding_side = "left"  # type: ignore

            # Load model with memory considerations
            model_kwargs = {
                "torch_dtype": self._torch_dtype,
                "attn_implementation": "eager",
                "low_cpu_mem_usage": True,
            }

            # Add memory constraints if specified
            if self._max_memory_mb:
                model_kwargs["max_memory"] = {0: f"{self._max_memory_mb}MB"}

            self._model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                self._model_name, **model_kwargs
            )
            if self._device != "auto":
                self._model = self._model.to(self._device)  # type: ignore
            self._model.eval()

            self.logger.info(f"HF model loaded on device: {self._device}")

        except Exception as e:
            self.logger.error(f"Failed to load HF model: {e}")
            raise

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens using the Hugging Face model."""
        try:
            with torch.no_grad():
                # Move input to device if needed
                if input_ids.device != torch.device(self._device):
                    input_ids = input_ids.to(self._device)

                # Create attention mask for proper padding handling
                attention_mask = torch.ones_like(input_ids)

                # Generate tokens
                outputs = self._model.generate(  # type: ignore
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id,  # type: ignore
                    eos_token_id=self._tokenizer.eos_token_id,  # type: ignore
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs,
                )

                # Extract generated token IDs (excluding input)
                generated_ids = outputs.sequences[  # type: ignore[union-attr]
                    :, input_ids.shape[1] :
                ]

                # Extract logits for the generated tokens
                if outputs.scores:  # type: ignore
                    logits = torch.stack(outputs.scores, dim=1)  # type: ignore
                else:
                    # Fallback: get logits from the last layer
                    with torch.no_grad():
                        last_hidden_states = self._model(input_ids).logits
                        logits = last_hidden_states[:, -max_new_tokens:, :]

                return generated_ids, logits

        except Exception as e:
            self.logger.error(f"Failed to generate tokens: {e}")
            raise

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer information for compatibility checking."""
        return {
            "model_name": self._model_name,
            "vocab_size": self._tokenizer.vocab_size,  # type: ignore
            "pad_token_id": self._tokenizer.pad_token_id,  # type: ignore
            "eos_token_id": self._tokenizer.eos_token_id,  # type: ignore
            "bos_token_id": self._tokenizer.bos_token_id,  # type: ignore
            "unk_token_id": self._tokenizer.unk_token_id,  # type: ignore
        }

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, return_tensors="pt")  # type: ignore

    def encode_with_attention_mask(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs with attention mask."""
        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def decode(self, token_ids) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, list):
            # Convert list to tensor and flatten if needed
            token_ids = torch.tensor(token_ids)
        elif not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)

        # Ensure we have a 1D tensor for decoding
        if token_ids.dim() > 1:
            token_ids = token_ids.flatten()

        return self._tokenizer.decode(  # type: ignore[union-attr]
            token_ids.tolist(), skip_special_tokens=True
        )

    @property
    def device(self) -> str:
        """Get the device this model is running on."""
        return self._device

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    @property
    def model(self):
        """Get the underlying model (read-only)."""
        return self._model

    def optimize(self, optimization_manager) -> None:
        """Optimize the model using the provided optimization manager."""
        if optimization_manager and hasattr(self, "_model") and self._model is not None:
            self._model = optimization_manager.optimize_model(self._model)

    def cleanup(self) -> None:
        """Clean up model memory (useful for MPS)."""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            del self._tokenizer

        # Force garbage collection
        import gc

        gc.collect()

        # Clear MPS cache if using MPS
        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.logger.info("Model memory cleaned up")


def create_tiny_hf_wrapper(
    model_name: str = "sshleifer/tiny-gpt2",
    device: str = "auto",
    max_memory_mb: int = 500,
    tokenizer: Optional[AutoTokenizer] = None,
) -> HFWrapper:
    """
    Create a Hugging Face wrapper with tiny model defaults for testing.

    Args:
        model_name: Tiny model name (default: sshleifer/tiny-gpt2)
        device: Device to run on
        max_memory_mb: Maximum memory usage in MB
        tokenizer: Shared tokenizer instance

    Returns:
        HFWrapper instance configured for tiny models
    """
    return HFWrapper(
        model_name=model_name,
        device=device,
        torch_dtype=None,  # Auto-select based on device
        max_memory_mb=max_memory_mb,
        tokenizer=tokenizer,
    )
