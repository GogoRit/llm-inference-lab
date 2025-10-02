"""
CPU/MPS Smoke Tests for LLM Inference Lab

Basic tests to validate the toolchain works on CPU/MPS without requiring CUDA.
These tests ensure torch and transformers are properly installed and functional.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def test_torch_import():
    """Test that PyTorch can be imported and basic functionality works."""
    assert torch is not None
    assert hasattr(torch, "version")
    print(f"PyTorch version: {torch.__version__}")


def test_device_availability():
    """Test device availability (CPU should always be available)."""
    assert torch.cuda.is_available() is not None  # Should be False on Mac
    assert (
        torch.backends.mps.is_available() is not None
    )  # Should be True on Apple Silicon

    # CPU should always be available
    cpu_device = torch.device("cpu")
    assert cpu_device.type == "cpu"


def test_tensor_operations():
    """Test basic tensor operations on CPU."""
    # Create a simple tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    # Basic operations
    z = x + y
    assert torch.allclose(z, torch.tensor([5.0, 7.0, 9.0]))

    # Matrix multiplication
    a = torch.randn(2, 3)
    b = torch.randn(3, 2)
    c = torch.mm(a, b)
    assert c.shape == (2, 2)


def test_transformers_import():
    """Test that transformers can be imported."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert AutoTokenizer is not None
        assert AutoModelForCausalLM is not None
        print("Transformers imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import transformers: {e}")


def test_local_baseline_import():
    """Test that our local baseline runner can be imported."""
    try:
        from src.server.local_baseline import LocalBaselineRunner

        assert LocalBaselineRunner is not None
        print("LocalBaselineRunner imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import LocalBaselineRunner: {e}")


def test_local_baseline_initialization():
    """Test that LocalBaselineRunner can be initialized (without loading model)."""
    try:
        from src.server.local_baseline import LocalBaselineRunner

        # This should work without actually loading the model
        # We'll test the device selection logic
        runner = LocalBaselineRunner.__new__(LocalBaselineRunner)
        device = runner._select_device()

        # Device should be one of the expected values
        assert device in ["cpu", "mps", "cuda"]
        print(f"Selected device: {device}")

    except Exception as e:
        pytest.fail(f"Failed to test LocalBaselineRunner initialization: {e}")


@pytest.mark.gpu
def test_mps_operations():
    """Test MPS operations if available (marked as GPU test)."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Test basic MPS operations
    device = torch.device("mps")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y

    assert z.device.type == "mps"
    assert torch.allclose(z.cpu(), torch.tensor([5.0, 7.0, 9.0]))


def test_model_loading_simulation():
    """Test that we can simulate model loading without actually downloading."""
    try:
        from transformers import AutoTokenizer

        # Test tokenizer initialization (this will download the model)
        # We'll use a very small model for this test
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

        # Test basic tokenization
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert len(tokens) > 0
        assert isinstance(decoded, str)
        print(f"Tokenization test passed: '{text}' -> {len(tokens)} tokens")

    except Exception as e:
        pytest.fail(f"Model loading simulation failed: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
