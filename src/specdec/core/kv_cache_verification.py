"""
KV Cache Verification for Debugging

Compares KV cache from target-only decoding vs speculative decoding
to ensure they match exactly.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def compute_kv_checksum(
    kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Compute checksum and sample of KV cache for verification.

    Args:
        kv_cache: KV cache tuple of (key, value) tuples per layer
        sample_size: Number of elements to sample for comparison

    Returns:
        Dictionary with checksums, shapes, and samples
    """
    if kv_cache is None or len(kv_cache) == 0:
        return {
            "exists": False,
            "num_layers": 0,
        }

    result: Dict[str, Any] = {
        "exists": True,
        "num_layers": len(kv_cache),
        "layers": {},
    }

    for layer_idx, (key, value) in enumerate(kv_cache):
        # Compute checksums
        key_checksum = torch.sum(key).item()
        value_checksum = torch.sum(value).item()

        # Sample first few elements
        key_sample = key.flatten()[:sample_size].tolist()
        value_sample = value.flatten()[:sample_size].tolist()

        result["layers"][layer_idx] = {
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "key_checksum": key_checksum,
            "value_checksum": value_checksum,
            "key_sample": key_sample,
            "value_sample": value_sample,
        }

    return result


def verify_kv_cache_alignment(
    target_only_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    speculative_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    tolerance: float = 1e-5,
) -> Tuple[bool, List[str]]:
    """
    Verify that target-only and speculative KV caches match.

    Args:
        target_only_kv: KV cache from target-only decoding
        speculative_kv: KV cache from speculative decoding
        tolerance: Numerical tolerance for comparison

    Returns:
        Tuple of (matches, list of error messages)
    """
    errors = []

    if target_only_kv is None and speculative_kv is None:
        return True, []

    if target_only_kv is None:
        errors.append("Target-only KV cache is None but speculative is not")
        return False, errors

    if speculative_kv is None:
        errors.append("Speculative KV cache is None but target-only is not")
        return False, errors

    if len(target_only_kv) != len(speculative_kv):
        errors.append(
            f"Number of layers mismatch: target={len(target_only_kv)}, "
            f"speculative={len(speculative_kv)}"
        )
        return False, errors

    for layer_idx, ((target_key, target_value), (spec_key, spec_value)) in enumerate(
        zip(target_only_kv, speculative_kv)
    ):
        # Check shapes
        if target_key.shape != spec_key.shape:
            errors.append(
                f"Layer {layer_idx} key shape mismatch: "
                f"target={target_key.shape}, speculative={spec_key.shape}"
            )

        if target_value.shape != spec_value.shape:
            errors.append(
                f"Layer {layer_idx} value shape mismatch: "
                f"target={target_value.shape}, speculative={spec_value.shape}"
            )

        # Check values (if shapes match)
        if target_key.shape == spec_key.shape:
            if not torch.allclose(target_key, spec_key, atol=tolerance):
                max_diff = torch.abs(target_key - spec_key).max().item()
                errors.append(
                    f"Layer {layer_idx} key values differ: max_diff={max_diff}"
                )

        if target_value.shape == spec_value.shape:
            if not torch.allclose(target_value, spec_value, atol=tolerance):
                max_diff = torch.abs(target_value - spec_value).max().item()
                errors.append(
                    f"Layer {layer_idx} value values differ: max_diff={max_diff}"
                )

    return len(errors) == 0, errors


def debug_verify_kv_cache_step(
    base_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    num_tokens: int,
    temperature: float,
    do_sample: bool,
    device: torch.device,
) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
    """
    Run one step of target-only decoding and return KV cache for verification.

    Args:
        base_model: Base language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        position_ids: Position IDs [batch_size, seq_len]
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        device: Device for tensors

    Returns:
        KV cache tuple after generation, or None if error
    """
    try:
        with torch.no_grad():
            # Generate tokens
            tokens, logits = base_model.generate_tokens(
                input_ids,
                max_new_tokens=num_tokens,
                temperature=temperature,
                do_sample=do_sample,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # Get KV cache
            if (
                hasattr(base_model, "_last_generated_kv_raw")
                and base_model._last_generated_kv_raw is not None
            ):
                return base_model._last_generated_kv_raw
            elif (
                hasattr(base_model, "_last_generated_kv")
                and base_model._last_generated_kv is not None
            ):
                if hasattr(base_model._last_generated_kv, "past_key_values"):
                    return base_model._last_generated_kv.past_key_values
                else:
                    return base_model._last_generated_kv

            return None
    except Exception as e:
        logger.error(f"Error in debug_verify_kv_cache_step: {e}")
        return None
