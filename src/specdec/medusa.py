"""
Medusa-lite Implementation for Speculative Decoding

This module implements a lightweight version of Medusa that uses multiple
small linear heads to predict multiple future tokens in parallel.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MedusaDraftor:
    """Medusa-lite draft model that uses multiple heads for parallel token prediction."""

    def __init__(
        self,
        base_model: Any,
        tokenizer: Any,
        num_heads: int = 2,
        head_init: str = "tie",
        temperature: float = 0.7,
        top_p: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize Medusa draftor.

        Args:
            base_model: Base Hugging Face model
            tokenizer: Hugging Face tokenizer
            num_heads: Number of prediction heads
            head_init: Head initialization method ("tie", "copy", "random")
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to run on
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.num_heads = num_heads
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

        # Get hidden size from base model
        if hasattr(base_model, "config"):
            self.hidden_size = base_model.config.hidden_size
        else:
            # Fallback for testing
            self.hidden_size = 768

        # Get vocabulary size
        if hasattr(tokenizer, "vocab_size"):
            self.vocab_size = tokenizer.vocab_size
        else:
            self.vocab_size = 1000  # Fallback for testing

        # Initialize prediction heads
        self.heads = self._create_heads(head_init)
        self.heads.to(device)

        logger.info(
            f"MedusaDraftor initialized: {num_heads} heads, "
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}"
        )

    def _create_heads(self, head_init: str) -> nn.ModuleList:
        """Create prediction heads with specified initialization."""
        heads = nn.ModuleList()
        base_lm_head = None

        # Get base model's language modeling head
        if hasattr(self.base_model, "lm_head"):
            base_lm_head = self.base_model.lm_head
        elif hasattr(self.base_model, "cls"):
            base_lm_head = self.base_model.cls

        for i in range(self.num_heads):
            head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            heads.append(head)

            # Initialize head weights
            if head_init == "tie" and base_lm_head is not None:
                # Share weights with base model's lm_head
                head.weight = base_lm_head.weight
            elif head_init == "copy" and base_lm_head is not None:
                # Copy weights from base model's lm_head
                with torch.no_grad():
                    head.weight.copy_(base_lm_head.weight)
            elif head_init == "random":
                # Random initialization (already done by default)
                pass
            else:
                logger.warning(
                    f"Unknown head_init '{head_init}', using random initialization"
                )

        return heads

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using Medusa heads.

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

            for step in range(max_new_tokens):
                step_logits = []
                step_tokens = []

                # Use each head to predict one token
                for head_idx in range(min(self.num_heads, max_new_tokens - step)):
                    head = self.heads[head_idx]
                    head_logits = head(current_hidden)  # [batch_size, 1, vocab_size]
                    step_logits.append(head_logits)

                    # Sample token from head logits
                    if self.temperature > 0:
                        head_logits = head_logits / self.temperature

                    # Apply top-p filtering
                    if self.top_p < 1.0:
                        head_logits = self._apply_top_p(head_logits, self.top_p)

                    # Sample token
                    probs = torch.softmax(head_logits, dim=-1)
                    next_token = torch.multinomial(
                        probs.squeeze(1), 1
                    )  # [batch_size, 1]
                    step_tokens.append(next_token)

                if not step_tokens:
                    break

                # Use first head's prediction for this step
                next_token = step_tokens[0]
                generated_ids.append(next_token)
                all_logits.append(step_logits[0])

                # Update hidden state for next iteration (simplified)
                # In a full implementation, we'd run the base model forward
                # For now, we'll use the current hidden state
                current_hidden = current_hidden  # Keep same hidden state

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
            "num_heads_used": len(step_tokens) if step_tokens else 0,
            "heads_per_step": [len(step_tokens) for _ in range(len(generated_ids))],
        }

        return generated_tokens, all_logits_tensor, generation_info

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) sampling to logits."""
        if top_p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff point
        cutoff = torch.searchsorted(cumulative_probs, top_p, right=False)

        # Create mask for tokens to keep
        mask = torch.arange(sorted_logits.size(-1), device=logits.device) < cutoff

        # Apply mask to sorted logits
        filtered_logits = torch.where(
            mask.unsqueeze(0).unsqueeze(0),
            sorted_logits,
            torch.full_like(sorted_logits, float("-inf")),
        )

        # Restore original order
        original_logits = torch.gather(filtered_logits, -1, sorted_indices.argsort(-1))

        return original_logits

    def get_info(self) -> Dict[str, Any]:
        """Get Medusa draftor information."""
        return {
            "type": "medusa",
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


def create_medusa_draftor(
    base_model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    device: str = "cpu",
) -> MedusaDraftor:
    """
    Create a MedusaDraftor from configuration.

    Args:
        base_model: Base Hugging Face model
        tokenizer: Hugging Face tokenizer
        config: Medusa configuration dictionary
        device: Device to run on

    Returns:
        Configured MedusaDraftor instance
    """
    medusa_config = config.get("medusa", {})

    return MedusaDraftor(
        base_model=base_model,
        tokenizer=tokenizer,
        num_heads=medusa_config.get("num_heads", 2),
        head_init=medusa_config.get("head_init", "tie"),
        temperature=medusa_config.get("temperature", 0.7),
        top_p=medusa_config.get("top_p", 1.0),
        device=device,
    )
