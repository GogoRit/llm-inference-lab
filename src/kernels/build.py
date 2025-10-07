"""
Build system for CUDA kernels with JIT compilation and caching.
"""

import hashlib
import logging
import os
from pathlib import Path

import torch
import torch.utils.cpp_extension

logger = logging.getLogger(__name__)

# Check if we should force Python fallback
FORCE_PYTHON = os.getenv("SPECDEC_FORCE_PY", "0").lower() in ("1", "true", "yes")


def get_cuda_arch():
    """Detect CUDA architecture for compilation."""
    if not torch.cuda.is_available():
        return None

    try:
        # Get CUDA capability
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}{capability[1]}"
        logger.info(f"Detected CUDA architecture: {arch}")
        return arch
    except Exception as e:
        logger.warning(f"Failed to detect CUDA architecture: {e}")
        return None


def get_file_hash(file_path):
    """Get SHA256 hash of file for caching."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_cuda_kernels():
    """Build CUDA kernels with caching."""
    if FORCE_PYTHON:
        logger.info("SPECDEC_FORCE_PY=1, skipping CUDA kernel compilation")
        return None

    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping kernel compilation")
        return None

    try:
        # Get paths
        kernels_dir = Path(__file__).parent
        cuda_dir = kernels_dir / "cuda"

        # Source files
        verify_cu = cuda_dir / "verify.cu"
        kv_cache_cu = cuda_dir / "kv_cache.cu"

        # Check if source files exist
        if not verify_cu.exists() or not kv_cache_cu.exists():
            logger.warning("CUDA source files not found, skipping compilation")
            return None

        # Get file hashes for caching
        verify_hash = get_file_hash(verify_cu)
        kv_cache_hash = get_file_hash(kv_cache_cu)

        # Create cache directory
        cache_dir = kernels_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)

        # Check if we have cached compiled kernels
        cuda_arch = get_cuda_arch()
        if cuda_arch:
            cache_file = (
                cache_dir
                / f"kernels_{cuda_arch}_{verify_hash[:8]}_{kv_cache_hash[:8]}.pt"
            )
            if cache_file.exists():
                logger.info(f"Loading cached kernels from {cache_file}")
                return torch.load(cache_file)

        # Compile kernels
        logger.info("Compiling CUDA kernels...")

        # Compile verify kernel
        verify_ext = torch.utils.cpp_extension.load(
            name="verify_cuda",
            sources=[str(verify_cu)],
            extra_cuda_cflags=[
                f"-arch=sm_{cuda_arch}" if cuda_arch else "-arch=sm_50",
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )

        # Compile KV cache kernel
        kv_cache_ext = torch.utils.cpp_extension.load(
            name="kv_cache_cuda",
            sources=[str(kv_cache_cu)],
            extra_cuda_cflags=[
                f"-arch=sm_{cuda_arch}" if cuda_arch else "-arch=sm_50",
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )

        # Cache compiled kernels
        if cuda_arch:
            compiled_kernels = {
                "verify_prefix": verify_ext.verify_prefix,
                "kv_append": kv_cache_ext.kv_append,
                "cuda_arch": cuda_arch,
                "verify_hash": verify_hash,
                "kv_cache_hash": kv_cache_hash,
            }
            torch.save(compiled_kernels, cache_file)
            logger.info(f"Cached compiled kernels to {cache_file}")

        return {
            "verify_prefix": verify_ext.verify_prefix,
            "kv_append": kv_cache_ext.kv_append,
            "cuda_arch": cuda_arch,
        }

    except Exception as e:
        logger.warning(f"Failed to compile CUDA kernels: {e}")
        return None


def load_kernels():
    """Load available kernels with fallback priority: CUDA -> Triton -> Python."""
    kernels = {}

    # Try CUDA first
    try:
        cuda_kernels = build_cuda_kernels()
        if cuda_kernels:
            kernels.update(cuda_kernels)
            kernels["verify_backend"] = "cuda"
            kernels["kv_append_backend"] = "cuda"
            logger.info("Using CUDA kernels")
            return kernels
    except Exception as e:
        logger.warning(f"CUDA kernels failed: {e}")

    # Try Triton fallback
    try:
        # import triton  # Unused import

        from .triton.verify import verify_prefix_triton

        kernels["verify_prefix"] = verify_prefix_triton
        kernels["verify_backend"] = "triton"
        kernels["kv_append_backend"] = "noop"  # Triton doesn't have KV append
        logger.info("Using Triton kernels (verify only)")
        return kernels
    except ImportError:
        logger.info("Triton not available, falling back to Python")
    except Exception as e:
        logger.warning(f"Triton kernels failed: {e}")

    # Fall back to Python reference implementation
    from . import reference

    kernels["verify_prefix"] = reference.verify_prefix_ref
    kernels["kv_append"] = reference.kv_append_ref
    kernels["verify_backend"] = "torch"
    kernels["kv_append_backend"] = "torch"
    logger.info("Using Python reference implementation")

    return kernels
