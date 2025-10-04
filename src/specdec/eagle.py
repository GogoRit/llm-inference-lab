"""
EAGLE-lite Implementation for Speculative Decoding

This module implements a lightweight version of EAGLE that uses hidden state
extrapolation to predict future tokens without additional model forward passes.
"""

import logging
from typing import Any, Dict, Tuple

import torch

logger = logging.getLogger(__name__)


class EagleDraftor:
    """EAGLE-lite draft model that uses hidden state extrapolation."""

    def __init__(
        self,
        base_model: Any,
        tokenizer: Any,
        alpha: float = 0.7,
        max_draft: int = 2,
        device: str = "cpu",
    ):
        """
        Initialize EAGLE draftor.

        Args:
            base_model: Base Hugging Face model
            tokenizer: Hugging Face tokenizer
            alpha: Extrapolation coefficient
            max_draft: Maximum number of draft tokens to generate
            device: Device to run on
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.max_draft = max_draft
        self.device = device

        # Get vocabulary size
        if hasattr(tokenizer, "vocab_size"):
            self.vocab_size = tokenizer.vocab_size
        else:
            self.vocab_size = 1000  # Fallback for testing

        # Get language modeling head
        if hasattr(base_model, "lm_head"):
            self.lm_head = base_model.lm_head
        elif hasattr(base_model, "cls"):
            self.lm_head = base_model.cls
        else:
            self.lm_head = None

        # State tracking
        self.last_hidden_states = None  # [batch_size, 2, hidden_size]
        self.last_tokens = None  # [batch_size, 2]

        logger.info(
            f"EagleDraftor initialized: alpha={alpha}, max_draft={max_draft}, "
            f"vocab_size={self.vocab_size}"
        )

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using EAGLE extrapolation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_ids, logits, generation_info)
        """
        start_time = (
            torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        )

        if start_time:
            start_time.record()

        batch_size, seq_len = input_ids.shape
        generated_ids = []
        all_logits = []

        # Get hidden states from base model
        with torch.no_grad():
            # Forward pass to get hidden states
            outputs = self.base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get last hidden state
            last_hidden_state = outputs.hidden_states[
                -1
            ]  # [batch_size, seq_len, hidden_size]
            current_hidden = last_hidden_state[
                :, -1:, :
            ]  # [batch_size, 1, hidden_size]

            # Update state tracking
            if self.last_hidden_states is None:
                # First call - initialize with current hidden state
                self.last_hidden_states = current_hidden.unsqueeze(
                    1
                )  # [batch_size, 1, hidden_size]
            else:
                # Update with new hidden state
                self.last_hidden_states = torch.cat(
                    [self.last_hidden_states[:, -1:, :], current_hidden.unsqueeze(1)],
                    dim=1,
                )  # [batch_size, 2, hidden_size]

            # Generate up to max_draft tokens
            num_tokens_to_generate = min(max_new_tokens, self.max_draft)

            for step in range(num_tokens_to_generate):
                # Extrapolate next hidden state
                if self.last_hidden_states.shape[1] >= 2:
                    # We have at least 2 hidden states for extrapolation
                    h_t_minus_1 = self.last_hidden_states[
                        :, -2, :
                    ]  # [batch_size, hidden_size]
                    h_t = self.last_hidden_states[:, -1, :]  # [batch_size, hidden_size]

                    # EAGLE extrapolation: h_next = h_t + alpha * (h_t - h_t_minus_1)
                    h_next = h_t + self.alpha * (h_t - h_t_minus_1)
                else:
                    # Fallback: use current hidden state
                    h_next = current_hidden.squeeze(1)  # [batch_size, hidden_size]

                # Get logits from language modeling head
                if self.lm_head is not None:
                    next_logits = self.lm_head(h_next)  # [batch_size, vocab_size]
                else:
                    # Fallback for testing
                    next_logits = torch.randn(
                        batch_size, self.vocab_size, device=self.device
                    )

                # Sample next token
                next_token = torch.argmax(
                    next_logits, dim=-1, keepdim=True
                )  # [batch_size, 1]

                generated_ids.append(next_token)
                all_logits.append(
                    next_logits.unsqueeze(1)
                )  # [batch_size, 1, vocab_size]

                # Update state for next iteration
                # In a full implementation, we'd run the base model forward
                # For now, we'll use extrapolation for the next hidden state
                h_next_expanded = h_next.unsqueeze(1)  # [batch_size, 1, hidden_size]
                self.last_hidden_states = torch.cat(
                    [self.last_hidden_states, h_next_expanded], dim=1
                )  # [batch_size, 3, hidden_size]

                # Keep only last 2 states for next extrapolation
                if self.last_hidden_states.shape[1] > 2:
                    self.last_hidden_states = self.last_hidden_states[:, -2:, :]

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            generation_time_ms = start_time.elapsed_time(end_time)
        else:
            generation_time_ms = 0.0

        # Concatenate generated tokens
        if generated_ids:
            generated_tokens = torch.cat(
                generated_ids, dim=1
            )  # [batch_size, num_generated]
            all_logits_tensor = torch.cat(
                all_logits, dim=1
            )  # [batch_size, num_generated, vocab_size]
        else:
            generated_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=self.device
            )
            all_logits_tensor = torch.empty(
                batch_size, 0, self.vocab_size, device=self.device
            )

        generation_info = {
            "generation_time_ms": generation_time_ms,
            "alpha": self.alpha,
            "extrapolation_steps": len(generated_ids),
            "state_history_length": (
                self.last_hidden_states.shape[1]
                if self.last_hidden_states is not None
                else 0
            ),
        }

        return generated_tokens, all_logits_tensor, generation_info

    def reset_state(self) -> None:
        """Reset internal state for new sequence."""
        self.last_hidden_states = None
        self.last_tokens = None

    def get_info(self) -> Dict[str, Any]:
        """Get EAGLE draftor information."""
        return {
            "type": "eagle",
            "alpha": self.alpha,
            "max_draft": self.max_draft,
            "vocab_size": self.vocab_size,
            "state_history_length": (
                self.last_hidden_states.shape[1]
                if self.last_hidden_states is not None
                else 0
            ),
        }


def create_eagle_draftor(
    base_model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    device: str = "cpu",
) -> EagleDraftor:
    """
    Create an EagleDraftor from configuration.

    Args:
        base_model: Base Hugging Face model
        tokenizer: Hugging Face tokenizer
        config: EAGLE configuration dictionary
        device: Device to run on

    Returns:
        Configured EagleDraftor instance
    """
    eagle_config = config.get("eagle", {})

    return EagleDraftor(
        base_model=base_model,
        tokenizer=tokenizer,
        alpha=eagle_config.get("alpha", 0.7),
        max_draft=eagle_config.get("max_draft", 2),
        device=device,
    )
