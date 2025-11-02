"""
Centralized Token Validation

Single efficient validation point to prevent invalid token IDs from reaching models.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def validate_and_clamp_tokens(
    input_ids: torch.Tensor,
    vocab_size: int,
    name: str = "input",
    strict: bool = False,
) -> torch.Tensor:
    """
    Efficiently validate and clamp token IDs to valid range (GPU-optimized).

    This is the ONLY validation point - all other validation code should be removed.
    All operations stay on GPU to minimize CPU overhead.

    Args:
        input_ids: Token tensor [batch, seq_len] or [seq_len] (kept on GPU)
        vocab_size: Model vocabulary size
        name: Identifier for logging
        strict: If True, sync to CPU to get min/max for detailed error messages

    Returns:
        Clamped token tensor (always valid: 0 <= token < vocab_size), stays on GPU
    """
    if input_ids is None or input_ids.numel() == 0:
        return input_ids

    # Fast GPU path: check if any invalid tokens exist (no CPU sync)
    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
    if invalid_mask.any():
        # For strict mode or significant corruption, get detailed diagnostics
        invalid_count = invalid_mask.sum()
        total_count = input_ids.numel()

        # Always get min/max when corruption detected (needed for debugging)
        # Move to CPU safely to avoid CUDA errors during error reporting
        try:
            min_val = input_ids.min().item()
            max_val = input_ids.max().item()
            invalid_count_cpu = invalid_count.item()

            # Log detailed error (as recommended in debugging guides)
            logger.error(
                f"[{name}] Input ID out of bounds detected! "
                f"Min: {min_val}, Max: {max_val}, Vocab_size: {vocab_size}, "
                f"Invalid_count: {invalid_count_cpu}/{total_count}"
            )
            print(
                f"[ERROR] [{name}] Input ID out of bounds: "
                f"Min={min_val}, Max={max_val}, Vocab_size={vocab_size}, "
                f"Invalid={invalid_count_cpu}/{total_count}",
                flush=True,
            )
        except Exception as e:
            # If tensor is corrupted and we can't read it, report that
            logger.error(f"[{name}] Tensor corrupted - cannot read min/max values: {e}")
            print(
                f"[ERROR] [{name}] Tensor corrupted - cannot read values. "
                f"Vocab_size={vocab_size}",
                flush=True,
            )

        # Clamp to valid range (efficient GPU operation, no CPU transfer)
        clamped = input_ids.clamp(min=0, max=vocab_size - 1)
        return clamped

    return input_ids


def get_vocab_size(model) -> Optional[int]:
    """Extract vocab_size from model config efficiently."""
    if hasattr(model, "_model") and hasattr(model._model, "config"):
        return getattr(model._model.config, "vocab_size", None)
    return None
