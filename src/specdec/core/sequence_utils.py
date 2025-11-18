"""
Sequence Utilities for Speculative Decoding

Provides helpers for padding, unpadding, and managing sequences in batches.
These utilities ensure consistent handling of variable-length sequences and
explicit position ID management.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def pad_sequences(
    sequences: List[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Pad a list of 1D tensors into a rectangular batch tensor.

    Args:
        sequences: List of 1D tensors [seq_len_i] with variable lengths
        pad_token_id: Token ID to use for padding
        device: Device for output tensors

    Returns:
        batch_tensor: [batch_size, max_len] padded batch tensor
        attention_mask: [batch_size, max_len] attention mask (1 for real tokens, 0 for padding)
        original_lengths: List of original sequence lengths
    """
    if not sequences:
        # Empty batch
        return (
            torch.empty(0, 0, dtype=torch.long, device=device),
            torch.empty(0, 0, dtype=torch.long, device=device),
            [],
        )

    # Get original lengths and max length
    original_lengths = [seq.shape[0] for seq in sequences]
    max_len = max(original_lengths)
    batch_size = len(sequences)

    # Pad each sequence to max_len
    padded_seqs: List[torch.Tensor] = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            pad_length = max_len - seq.shape[0]
            # Pad on the right: (left, right) for 1D tensor
            seq_padded = F.pad(
                seq,
                (0, pad_length),
                value=pad_token_id,
                mode="constant",
            )
            padded_seqs.append(seq_padded)
        else:
            padded_seqs.append(seq)

    # Stack into batch tensor
    batch_tensor = torch.stack(padded_seqs, dim=0).contiguous()  # [batch_size, max_len]

    # Create attention mask: 1 for real tokens, 0 for padding
    attention_mask = torch.ones(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )
    for i, orig_len in enumerate(original_lengths):
        if orig_len < max_len:
            attention_mask[i, orig_len:] = 0

    return batch_tensor, attention_mask, original_lengths


def unpad_sequences(
    batch_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Unpad a batch tensor back into a list of 1D tensors.

    Args:
        batch_tensor: [batch_size, max_len] padded batch tensor
        attention_mask: [batch_size, max_len] attention mask (1 for real tokens, 0 for padding)

    Returns:
        sequences: List of 1D tensors [seq_len_i] with original lengths
    """
    batch_size = batch_tensor.shape[0]
    sequences = []

    for i in range(batch_size):
        # Get actual length from attention mask
        actual_length = int(attention_mask[i].sum().item())
        # Extract sequence without padding
        seq = batch_tensor[i, :actual_length]
        sequences.append(seq)

    return sequences


def unpad_append_repad(
    sequences: List[torch.Tensor],
    tokens_to_append: List[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Complete unpad → append → repad cycle in a single operation.

    This is the core operation in speculative decoding:
    1. Sequences are already unpadded (list of 1D tensors)
    2. Append tokens_to_append[i] to sequences[i] for each i
    3. Repad to create a new rectangular batch

    Args:
        sequences: List of 1D tensors [seq_len_i] (already unpadded)
        tokens_to_append: List of 1D tensors [append_len_i] to append to each sequence
        pad_token_id: Token ID to use for padding
        device: Device for output tensors

    Returns:
        batch_tensor: [batch_size, max_len] new padded batch tensor
        attention_mask: [batch_size, max_len] attention mask
        original_lengths: List of new sequence lengths after appending
    """
    if len(sequences) != len(tokens_to_append):
        raise ValueError(
            f"Number of sequences ({len(sequences)}) must match "
            f"number of token lists to append ({len(tokens_to_append)})"
        )

    # Append tokens to each sequence
    updated_sequences = []
    for seq, tokens in zip(sequences, tokens_to_append):
        if tokens.shape[0] > 0:
            updated_seq = torch.cat([seq, tokens], dim=0)
        else:
            updated_seq = seq
        updated_sequences.append(updated_seq)

    # Repad the updated sequences
    return pad_sequences(updated_sequences, pad_token_id, device)


def create_position_ids(
    sequence_lengths: List[int],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create position IDs starting from 0 for the first non-padding token of each sequence.

    Position IDs are monotone and contiguous for non-padding tokens:
    - Position 0 for first token
    - Position 1 for second token
    - ...
    - Position (length-1) for last token
    - Padding positions can be 0 (will be masked by attention mask)

    Args:
        sequence_lengths: List of actual sequence lengths (excluding padding)
        max_length: Maximum sequence length in batch
        device: Device for output tensor

    Returns:
        position_ids: [batch_size, max_length] tensor with positions starting from 0
    """
    batch_size = len(sequence_lengths)
    position_ids = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device,
    )

    for i, length in enumerate(sequence_lengths):
        # Create positions 0, 1, 2, ..., length-1 for this sequence
        position_ids[i, :length] = torch.arange(length, device=device)
        # Padding positions remain 0 (will be masked by attention mask)

    return position_ids
