#!/usr/bin/env python3
"""
Tests for sequence utilities (padding, unpadding, position IDs).

Verifies that:
1. Unpad then repad is lossless
2. Position IDs are monotone and contiguous per sequence for non-padding tokens
3. Attention masks correctly exclude padding
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch

from specdec.core.sequence_utils import (
    create_position_ids,
    pad_sequences,
    unpad_append_repad,
    unpad_sequences,
)


def test_pad_unpad_lossless():
    """Test that unpad then repad is lossless."""
    device = torch.device("cpu")
    pad_token_id = 0

    # Create toy sequences with different lengths
    seq1 = torch.tensor([1, 2, 3], dtype=torch.long)
    seq2 = torch.tensor([4, 5], dtype=torch.long)
    seq3 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.long)

    sequences = [seq1, seq2, seq3]
    original_lengths = [seq.shape[0] for seq in sequences]

    # Pad sequences
    batch_tensor, attention_mask, lengths = pad_sequences(
        sequences=sequences,
        pad_token_id=pad_token_id,
        device=device,
    )

    # Verify lengths match
    assert (
        lengths == original_lengths
    ), f"Lengths mismatch: {lengths} != {original_lengths}"

    # Unpad sequences
    unpadded_sequences = unpad_sequences(batch_tensor, attention_mask)

    # Verify we get back the original sequences
    assert len(unpadded_sequences) == len(sequences), "Number of sequences mismatch"

    for i, (original, unpadded) in enumerate(zip(sequences, unpadded_sequences)):
        assert torch.equal(original, unpadded), (
            f"Sequence {i} mismatch:\n"
            f"  Original: {original.tolist()}\n"
            f"  Unpadded: {unpadded.tolist()}"
        )

    print("PASSED: Unpad then repad is lossless")


def test_unpad_append_repad():
    """Test the unpad → append → repad cycle."""
    device = torch.device("cpu")
    pad_token_id = 0

    # Start with sequences
    seq1 = torch.tensor([1, 2, 3], dtype=torch.long)
    seq2 = torch.tensor([4, 5], dtype=torch.long)
    seq3 = torch.tensor([6, 7, 8], dtype=torch.long)
    sequences = [seq1, seq2, seq3]

    # Tokens to append
    append1 = torch.tensor([10, 11], dtype=torch.long)
    append2 = torch.tensor([12], dtype=torch.long)
    append3 = torch.tensor([13, 14, 15, 16], dtype=torch.long)
    tokens_to_append = [append1, append2, append3]

    # Expected results after append
    expected1 = torch.tensor([1, 2, 3, 10, 11], dtype=torch.long)
    expected2 = torch.tensor([4, 5, 12], dtype=torch.long)
    expected3 = torch.tensor([6, 7, 8, 13, 14, 15, 16], dtype=torch.long)
    expected_sequences = [expected1, expected2, expected3]

    # Perform unpad → append → repad
    batch_tensor, attention_mask, lengths = unpad_append_repad(
        sequences=sequences,
        tokens_to_append=tokens_to_append,
        pad_token_id=pad_token_id,
        device=device,
    )

    # Unpad to verify
    result_sequences = unpad_sequences(batch_tensor, attention_mask)

    # Verify results
    assert len(result_sequences) == len(
        expected_sequences
    ), "Number of sequences mismatch"

    for i, (expected, result) in enumerate(zip(expected_sequences, result_sequences)):
        assert torch.equal(expected, result), (
            f"Sequence {i} mismatch after append:\n"
            f"  Expected: {expected.tolist()}\n"
            f"  Result: {result.tolist()}"
        )

    # Verify lengths
    expected_lengths = [seq.shape[0] for seq in expected_sequences]
    assert (
        lengths == expected_lengths
    ), f"Lengths mismatch: {lengths} != {expected_lengths}"

    print("PASSED: Unpad -> append -> repad works correctly")


def test_position_ids_monotone_contiguous():
    """Test that position IDs are monotone and contiguous per sequence."""
    device = torch.device("cpu")

    # Test case 1: Different length sequences
    sequence_lengths = [3, 5, 2]
    max_length = 5

    position_ids = create_position_ids(
        sequence_lengths=sequence_lengths,
        max_length=max_length,
        device=device,
    )

    # Verify shape
    assert position_ids.shape == (3, 5), f"Wrong shape: {position_ids.shape}"

    # Verify each sequence has monotone, contiguous positions
    for i, length in enumerate(sequence_lengths):
        seq_positions = position_ids[i, :length].tolist()
        expected_positions = list(range(length))
        assert seq_positions == expected_positions, (
            f"Sequence {i} positions not contiguous:\n"
            f"  Got: {seq_positions}\n"
            f"  Expected: {expected_positions}"
        )

        # Verify padding positions are 0 (or any value, but will be masked)
        if length < max_length:
            padding_positions = position_ids[i, length:].tolist()
            # Padding positions should all be 0 (implementation detail)
            assert all(
                p == 0 for p in padding_positions
            ), f"Sequence {i} padding positions not zero: {padding_positions}"

    # Test case 2: All same length
    sequence_lengths = [4, 4, 4]
    max_length = 4

    position_ids = create_position_ids(
        sequence_lengths=sequence_lengths,
        max_length=max_length,
        device=device,
    )

    for i in range(3):
        seq_positions = position_ids[i, :].tolist()
        expected_positions = [0, 1, 2, 3]
        assert (
            seq_positions == expected_positions
        ), f"Sequence {i} positions not correct: {seq_positions}"

    # Test case 3: Single sequence
    sequence_lengths = [7]
    max_length = 7

    position_ids = create_position_ids(
        sequence_lengths=sequence_lengths,
        max_length=max_length,
        device=device,
    )

    seq_positions = position_ids[0, :].tolist()
    expected_positions = [0, 1, 2, 3, 4, 5, 6]
    assert (
        seq_positions == expected_positions
    ), f"Single sequence positions not correct: {seq_positions}"

    print("PASSED: Position IDs are monotone and contiguous")


def test_attention_mask_excludes_padding():
    """Test that attention masks correctly exclude padding tokens."""
    device = torch.device("cpu")
    pad_token_id = 0

    # Create sequences with different lengths
    seq1 = torch.tensor([1, 2, 3], dtype=torch.long)
    seq2 = torch.tensor([4, 5], dtype=torch.long)
    seq3 = torch.tensor([6, 7, 8, 9], dtype=torch.long)
    sequences = [seq1, seq2, seq3]

    # Pad sequences
    batch_tensor, attention_mask, lengths = pad_sequences(
        sequences=sequences,
        pad_token_id=pad_token_id,
        device=device,
    )

    max_length = batch_tensor.shape[1]

    # Verify attention mask
    for i, length in enumerate(lengths):
        # Real tokens should have mask = 1
        real_mask = attention_mask[i, :length].tolist()
        assert all(
            m == 1 for m in real_mask
        ), f"Sequence {i}: Real tokens should have mask=1, got {real_mask}"

        # Padding tokens should have mask = 0
        if length < max_length:
            padding_mask = attention_mask[i, length:].tolist()
            assert all(
                m == 0 for m in padding_mask
            ), f"Sequence {i}: Padding tokens should have mask=0, got {padding_mask}"

    # Verify batch tensor has padding tokens
    for i, length in enumerate(lengths):
        if length < max_length:
            padding_tokens = batch_tensor[i, length:].tolist()
            assert all(
                t == pad_token_id for t in padding_tokens
            ), f"Sequence {i}: Padding tokens should be {pad_token_id}, got {padding_tokens}"

    print("PASSED: Attention masks correctly exclude padding")


def test_empty_batch():
    """Test handling of empty batch."""
    device = torch.device("cpu")
    pad_token_id = 0

    # Empty batch
    batch_tensor, attention_mask, lengths = pad_sequences(
        sequences=[],
        pad_token_id=pad_token_id,
        device=device,
    )

    assert batch_tensor.shape == (
        0,
        0,
    ), f"Wrong shape for empty batch: {batch_tensor.shape}"
    assert attention_mask.shape == (0, 0), f"Wrong mask shape: {attention_mask.shape}"
    assert lengths == [], f"Wrong lengths: {lengths}"

    # Unpad empty batch
    unpadded = unpad_sequences(batch_tensor, attention_mask)
    assert unpadded == [], f"Unpadded empty batch should be empty: {unpadded}"

    print("PASSED: Empty batch handled correctly")


def test_single_sequence():
    """Test handling of single sequence (no padding needed)."""
    device = torch.device("cpu")
    pad_token_id = 0

    seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    sequences = [seq]

    batch_tensor, attention_mask, lengths = pad_sequences(
        sequences=sequences,
        pad_token_id=pad_token_id,
        device=device,
    )

    assert batch_tensor.shape == (1, 5), f"Wrong shape: {batch_tensor.shape}"
    assert attention_mask.shape == (1, 5), f"Wrong mask shape: {attention_mask.shape}"
    assert lengths == [5], f"Wrong lengths: {lengths}"

    # Verify all tokens are real (no padding)
    assert attention_mask[0].sum().item() == 5, "All tokens should be real"

    # Unpad
    unpadded = unpad_sequences(batch_tensor, attention_mask)
    assert len(unpadded) == 1, "Should have one sequence"
    assert torch.equal(unpadded[0], seq), "Unpadded sequence should match original"

    print("PASSED: Single sequence handled correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("SEQUENCE UTILITIES TESTS")
    print("=" * 80)
    print()

    try:
        test_pad_unpad_lossless()
        test_unpad_append_repad()
        test_position_ids_monotone_contiguous()
        test_attention_mask_excludes_padding()
        test_empty_batch()
        test_single_sequence()

        print()
        print("=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        return True
    except AssertionError as e:
        print()
        print("=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
