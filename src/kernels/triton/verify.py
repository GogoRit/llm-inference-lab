"""
Triton implementation of verify_prefix kernel as fallback for CUDA.
Provides identical signature and behavior to CUDA kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def verify_prefix_triton_kernel(
    logits_ptr,  # [B][K][V]
    draft_ids_ptr,  # [B][K]
    accept_len_ptr,  # [B]
    accepted_mask_ptr,  # [B][K]
    B,
    K,
    V,
    logits_stride_b,
    logits_stride_k,
    logits_stride_v,
    draft_ids_stride_b,
    draft_ids_stride_k,
    accepted_mask_stride_b,
    accepted_mask_stride_k,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    if batch_idx >= B:
        return

    # Process each K position
    for k in range(K):
        # Load draft_id for this position
        draft_id_offset = batch_idx * draft_ids_stride_b + k * draft_ids_stride_k
        target_id = tl.load(draft_ids_ptr + draft_id_offset)

        # Find argmax in logits[batch_idx][k][:]
        # CRITICAL FIX: Initialize max_val by loading first value to match dtype (fp16/fp32)
        # This prevents type mismatch errors when logits are fp16
        # Using the first logit value ensures perfect dtype matching
        first_logit_offset = batch_idx * logits_stride_b + k * logits_stride_k
        max_val = tl.load(logits_ptr + first_logit_offset)
        argmax_idx = 0

        # Process remaining values (skip v=0 since we already loaded it as initial max_val)
        # Edge case: If V == 1, this loop won't execute any iterations, which is correct
        for v_start in range(0, V, BLOCK_SIZE):
            v_end = min(v_start + BLOCK_SIZE, V)
            v_range = v_end - v_start

            # Load logits for this chunk
            logits_offset = (
                batch_idx * logits_stride_b
                + k * logits_stride_k
                + v_start * logits_stride_v
            )

            # Vectorized argmax within chunk
            # For first chunk (v_start=0), skip v=0 since we already loaded it
            # For subsequent chunks, process all values
            start_v = 1 if v_start == 0 else 0
            for v in range(start_v, v_range):
                val = tl.load(logits_ptr + logits_offset + v * logits_stride_v)
                if val > max_val:
                    max_val = val
                    argmax_idx = v_start + v

        # Check if argmax matches target_id
        is_match = argmax_idx == target_id

        # Store result in accepted_mask
        mask_offset = batch_idx * accepted_mask_stride_b + k * accepted_mask_stride_k
        tl.store(accepted_mask_ptr + mask_offset, is_match.to(tl.uint8))

    # Compute prefix length (sequential check)
    prefix_len = 0
    for k in range(K):
        mask_offset = batch_idx * accepted_mask_stride_b + k * accepted_mask_stride_k
        is_accepted = tl.load(accepted_mask_ptr + mask_offset)

        if is_accepted:
            prefix_len += 1
        else:
            break

    # Store accept_len
    accept_len_offset = batch_idx
    tl.store(accept_len_ptr + accept_len_offset, prefix_len)


def verify_prefix_triton(
    logits: torch.Tensor, draft_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of verify_prefix.

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
    assert logits.device.type == "cuda", "logits must be on CUDA"
    assert draft_ids.device.type == "cuda", "draft_ids must be on CUDA"

    B, K, V = logits.shape

    # Create output tensors
    accept_len = torch.zeros(B, dtype=torch.int32, device=logits.device)
    accepted_mask = torch.zeros((B, K), dtype=torch.uint8, device=logits.device)

    # Get strides
    logits_stride_b, logits_stride_k, logits_stride_v = logits.stride()
    draft_ids_stride_b, draft_ids_stride_k = draft_ids.stride()
    accepted_mask_stride_b, accepted_mask_stride_k = accepted_mask.stride()

    # Launch kernel
    BLOCK_SIZE = 1024  # Process V in chunks of 1024
    grid = (B,)

    verify_prefix_triton_kernel[grid](
        logits,
        draft_ids,
        accept_len,
        accepted_mask,
        B,
        K,
        V,
        logits_stride_b,
        logits_stride_k,
        logits_stride_v,
        draft_ids_stride_b,
        draft_ids_stride_k,
        accepted_mask_stride_b,
        accepted_mask_stride_k,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # CRITICAL FIX: Synchronize to catch kernel errors early
    # This ensures any dtype mismatch or other errors are caught before returning
    torch.cuda.synchronize()

    return accept_len, accepted_mask
