"""
Reference implementation of kernels in pure PyTorch.
Used as fallback when CUDA/Triton kernels are not available.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def verify_prefix_ref(
    logits: torch.Tensor, draft_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of verify_prefix in pure PyTorch.

    Args:
        logits: [B][K][V] tensor of logits
        draft_ids: [B][K] tensor of draft token IDs

    Returns:
        accept_len: [B] tensor of accepted prefix lengths
        accepted_mask: [B][K] tensor of accepted positions (1 for accepted, 0 for rejected)
    """
    # Input validation
    assert logits.dim() == 3, "logits must be 3D tensor [B][K][V]"
    assert draft_ids.dim() == 2, "draft_ids must be 2D tensor [B][K]"
    assert logits.size(0) == draft_ids.size(0), "Batch size mismatch"
    assert logits.size(1) == draft_ids.size(1), "K dimension mismatch"

    B, K, V = logits.shape
    device = logits.device

    # Find argmax for each position
    predicted_ids = torch.argmax(logits, dim=-1)  # [B][K]

    # Check matches
    matches = predicted_ids == draft_ids  # [B][K]

    # Compute prefix lengths
    accept_len = torch.zeros(B, dtype=torch.int32, device=device)
    accepted_mask = torch.zeros((B, K), dtype=torch.uint8, device=device)

    for b in range(B):
        prefix_len = 0
        for k in range(K):
            if matches[b, k]:
                prefix_len += 1
                accepted_mask[b, k] = 1
            else:
                break
        accept_len[b] = prefix_len

    return accept_len, accepted_mask


def kv_append_ref(
    base_k: torch.Tensor,
    base_v: torch.Tensor,
    draft_k: torch.Tensor,
    draft_v: torch.Tensor,
    accepted_mask: torch.Tensor,
    accept_len: torch.Tensor,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of KV cache append in pure PyTorch.

    Args:
        base_k: [B][H][L][D] base key cache
        base_v: [B][H][L][D] base value cache
        draft_k: [B][H][K][D] draft key cache
        draft_v: [B][H][K][D] draft value cache
        accepted_mask: [B][K] accepted positions
        accept_len: [B] number of accepted tokens per batch
        offset: where to start appending in base cache

    Returns:
        output_k: [B][H][L+K][D] output key cache
        output_v: [B][H][L+K][D] output value cache
    """
    # Input validation
    assert base_k.dim() == 4, "base_k must be 4D tensor [B][H][L][D]"
    assert base_v.dim() == 4, "base_v must be 4D tensor [B][H][L][D]"
    assert draft_k.dim() == 4, "draft_k must be 4D tensor [B][H][K][D]"
    assert draft_v.dim() == 4, "draft_v must be 4D tensor [B][H][K][D]"
    assert accepted_mask.dim() == 2, "accepted_mask must be 2D tensor [B][K]"
    assert accept_len.dim() == 1, "accept_len must be 1D tensor [B]"

    B, H, L, D = base_k.shape
    K = draft_k.shape[2]
    device = base_k.device

    # Create output tensors
    output_k = torch.zeros((B, H, L + K, D), dtype=base_k.dtype, device=device)
    output_v = torch.zeros((B, H, L + K, D), dtype=base_v.dtype, device=device)

    # Copy base cache
    output_k[:, :, :L, :] = base_k
    output_v[:, :, :L, :] = base_v

    # Append accepted draft positions
    for b in range(B):
        num_accepted = accept_len[b].item()
        if num_accepted == 0:
            continue

        accepted_count = 0
        for k in range(K):
            if accepted_mask[b, k]:
                output_k[b, :, L + accepted_count, :] = draft_k[b, :, k, :]
                output_v[b, :, L + accepted_count, :] = draft_v[b, :, k, :]
                accepted_count += 1
                if accepted_count >= num_accepted:
                    break

    return output_k, output_v
