"""
Basic tests for LLM Inference Lab

These are placeholder tests to ensure the CI pipeline runs successfully.
More comprehensive tests will be added as the project develops.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def test_project_structure():
    """Test that the basic project structure exists."""
    assert PROJECT_ROOT.exists()
    assert SRC_DIR.exists()
    assert (PROJECT_ROOT / "tests").exists()
    assert (PROJECT_ROOT / "configs").exists()
    assert (PROJECT_ROOT / "scripts").exists()
    assert (PROJECT_ROOT / "docs").exists()


def test_src_modules_importable():
    """Test that src modules can be imported."""
    try:
        import src

        assert src.__version__ == "0.1.0"
    except ImportError as e:
        pytest.fail(f"Failed to import src module: {e}")


def test_requirements_file_exists():
    """Test that requirements.txt exists."""
    requirements_file = PROJECT_ROOT / "env" / "requirements.txt"
    assert requirements_file.exists()
    assert requirements_file.stat().st_size > 0


@pytest.mark.gpu
def test_gpu_availability():
    """Test GPU availability (marked as GPU test)."""
    try:
        import torch

        if torch.cuda.is_available():
            assert torch.cuda.device_count() > 0
        else:
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_cpu_only():
    """Test that runs on CPU only."""
    assert True  # Placeholder test


def test_integration():
    """Test integration functionality."""
    assert True  # Placeholder test
