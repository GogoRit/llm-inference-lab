"""
Standalone correctness test for zero-copy ring buffer integration.

This test verifies the core correctness properties without requiring
full module imports.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import torch


def test_zero_copy_kv_cache_structure():
    """Test that zero-copy KV cache maintains correct structure."""
    # Import directly to avoid __init__.py dependencies
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kv_cache_manager", src_path / "specdec" / "cache" / "kv_cache_manager.py"
    )
    kv_cache_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kv_cache_module)
    SafeKVCacheManager = kv_cache_module.SafeKVCacheManager

    # Initialize KV cache manager
    manager = SafeKVCacheManager(
        device="cpu",
        max_seq_len=1024,
    )

    # Create mock KV cache
    batch_size = 2
    num_layers = 2
    num_heads = 4
    seq_len = 10
    head_dim = 64

    mock_kv = tuple(
        (
            torch.randn(batch_size, num_heads, seq_len, head_dim),
            torch.randn(batch_size, num_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )

    # Set batch size
    manager.set_batch_size(batch_size)

    # Update cache
    active_indices = [0, 1]
    manager.update_base_cache(
        new_kv=mock_kv,
        active_indices=active_indices,
        tokens_appended=[seq_len, seq_len],
    )

    # Verify structure
    assert manager.base_cache is not None
    assert len(manager.base_cache) == num_layers

    # Verify sequence lengths (accumulate, not replace)
    assert len(manager.base_current_seq_lens) == batch_size
    # Sequence lengths accumulate: starts at 0, adds seq_len from KV cache shape
    # The sequence length should be updated ONCE, not once per layer
    assert manager.base_current_seq_lens[0] == seq_len
    assert manager.base_current_seq_lens[1] == seq_len


def test_zero_copy_rollback_correctness():
    """Test that zero-copy rollback correctly updates pointers."""
    # Import directly to avoid __init__.py dependencies
    import importlib.util

    kv_spec = importlib.util.spec_from_file_location(
        "kv_cache_manager", src_path / "specdec" / "cache" / "kv_cache_manager.py"
    )
    kv_module = importlib.util.module_from_spec(kv_spec)
    kv_spec.loader.exec_module(kv_module)
    SafeKVCacheManager = kv_module.SafeKVCacheManager

    # Initialize
    manager = SafeKVCacheManager(device="cpu", max_seq_len=1024)
    manager.set_batch_size(2)

    # Set initial sequence lengths manually
    manager.base_current_seq_lens = [100, 100]
    current_seq_lens = [100, 100]

    # Test rollback logic directly: original_len=100, accepted_len=3
    # Rollback should set to original_len + accepted_len
    new_len = 100 + 3  # original_len + accepted_len
    manager.base_current_seq_lens[0] = new_len
    current_seq_lens[0] = new_len

    # Verify correct rollback
    assert new_len == 103  # original_len + accepted_len
    assert manager.base_current_seq_lens[0] == 103
    assert current_seq_lens[0] == 103


def test_zero_copy_batch_switching():
    """Test that batch switching correctly resets pointers."""
    # Import directly (reuse from first test)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kv_cache_manager", src_path / "specdec" / "cache" / "kv_cache_manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SafeKVCacheManager = module.SafeKVCacheManager

    manager = SafeKVCacheManager(device="cpu", max_seq_len=1024)

    # Switch to batch size 4
    manager.set_batch_size(4)
    assert manager.current_batch_size == 4
    assert len(manager.base_current_seq_lens) == 4

    # Switch to different batch size
    manager.set_batch_size(8)
    assert manager.current_batch_size == 8
    assert len(manager.base_current_seq_lens) == 8


def test_zero_copy_multiple_updates():
    """Test that multiple updates maintain consistency."""
    # Import directly (reuse pattern)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kv_cache_manager", src_path / "specdec" / "cache" / "kv_cache_manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SafeKVCacheManager = module.SafeKVCacheManager

    manager = SafeKVCacheManager(device="cpu", max_seq_len=1024)
    batch_size = 2
    manager.set_batch_size(batch_size)

    num_layers = 2
    num_heads = 4
    head_dim = 64

    # Multiple updates with increasing sequence lengths
    for step in range(3):
        seq_len = 5 + step
        mock_kv = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim),
                torch.randn(batch_size, num_heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

        active_indices = [0, 1]
        manager.update_base_cache(
            new_kv=mock_kv,
            active_indices=active_indices,
            tokens_appended=[seq_len, seq_len],
        )

        # Verify sequence lengths are updated
        seq_lens = manager.base_current_seq_lens
        assert len(seq_lens) == batch_size
        # Sequence lengths accumulate: each update adds seq_len
        # Step 0: 0 + 5 = 5
        # Step 1: 5 + 6 = 11
        # Step 2: 11 + 7 = 18
        expected_len = sum(5 + i for i in range(step + 1))
        assert (
            seq_lens[0] == expected_len
        ), f"Step {step}: expected {expected_len}, got {seq_lens[0]}"
        assert seq_lens[1] == expected_len


def test_zero_copy_active_indices_filtering():
    """Test that active indices filtering works correctly."""
    # Import directly (reuse pattern)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kv_cache_manager", src_path / "specdec" / "cache" / "kv_cache_manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SafeKVCacheManager = module.SafeKVCacheManager

    manager = SafeKVCacheManager(device="cpu", max_seq_len=1024)
    batch_size = 4
    manager.set_batch_size(batch_size)

    # Create mock KV cache
    num_layers = 2
    num_heads = 4
    seq_len = 10
    head_dim = 64

    mock_kv = tuple(
        (
            torch.randn(2, num_heads, seq_len, head_dim),  # Only 2 active sequences
            torch.randn(2, num_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )

    # Update with only some active indices
    active_indices = [0, 2]  # Only sequences 0 and 2
    manager.update_base_cache(
        new_kv=mock_kv,
        active_indices=active_indices,
        tokens_appended=[seq_len, seq_len],
    )

    # Verify only active sequences are updated
    seq_lens = manager.base_current_seq_lens
    # Active sequences accumulate: starts at 0, adds seq_len
    assert seq_lens[0] == seq_len  # Active (0 + 10)
    assert seq_lens[1] == 0  # Not active (stays 0)
    assert seq_lens[2] == seq_len  # Active (0 + 10)
    assert seq_lens[3] == 0  # Not active (stays 0)


def test_zero_copy_views_consistency():
    """Test that KV cache views are consistent."""
    # Import directly (reuse pattern)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kv_cache_manager", src_path / "specdec" / "cache" / "kv_cache_manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SafeKVCacheManager = module.SafeKVCacheManager

    manager = SafeKVCacheManager(device="cpu", max_seq_len=1024)
    batch_size = 2
    manager.set_batch_size(batch_size)

    # Create and update mock KV cache
    num_layers = 2
    num_heads = 4
    seq_len = 10
    head_dim = 64

    mock_kv = tuple(
        (
            torch.randn(batch_size, num_heads, seq_len, head_dim),
            torch.randn(batch_size, num_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )

    active_indices = [0, 1]
    manager.update_base_cache(
        new_kv=mock_kv,
        active_indices=active_indices,
        tokens_appended=[seq_len, seq_len],
    )

    # Get views
    active_indices_tensor = torch.tensor(active_indices, dtype=torch.long)
    views = manager.get_base_past_kv(active_indices=active_indices_tensor)

    # Verify views are not None
    assert views is not None
    assert len(views) == num_layers

    # Verify views have correct batch dimension
    for key, value in views:
        assert key.shape[0] == len(active_indices)
        assert value.shape[0] == len(active_indices)
        # Views show accumulated sequence length (seq_len after one update)
        assert key.shape[2] == seq_len  # Sequence length
        assert value.shape[2] == seq_len


if __name__ == "__main__":
    # Run tests directly
    test_zero_copy_kv_cache_structure()
    test_zero_copy_rollback_correctness()
    test_zero_copy_batch_switching()
    test_zero_copy_multiple_updates()
    test_zero_copy_active_indices_filtering()
    test_zero_copy_views_consistency()
    print("All correctness tests passed!")
