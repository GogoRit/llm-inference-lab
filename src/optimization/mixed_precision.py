"""
Mixed Precision and Gradient Checkpointing Optimizations

Provides CPU/MPS optimized mixed precision support and gradient checkpointing
for improved memory efficiency and performance in local development.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


class _NoopContext:
    """No-op context manager for when mixed precision is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class MixedPrecisionManager:
    """Manages mixed precision operations for CPU/MPS optimization."""

    def __init__(
        self,
        device: str = "auto",
        enabled: bool = True,
        dtype: Optional[torch.dtype] = None,
        memory_efficient: bool = True,
    ):
        """
        Initialize mixed precision manager.

        Args:
            device: Device to run on ("auto", "cpu", "mps", "cuda")
            enabled: Whether mixed precision is enabled
            dtype: Specific dtype to use (overrides device-based selection)
            memory_efficient: Whether to use memory-efficient attention
        """
        self.device = self._select_device(device)

        # Check environment overrides
        env_amp = os.getenv("SPECDEC_AMP")
        if env_amp is not None:
            enabled = env_amp.lower() in ("1", "true", "yes")

        env_dtype = os.getenv("SPECDEC_DTYPE")
        if env_dtype is not None:
            if env_dtype.lower() == "float16":
                dtype = torch.float16
            elif env_dtype.lower() == "bfloat16":
                dtype = torch.bfloat16
            elif env_dtype.lower() == "float32":
                dtype = torch.float32

        self.enabled = enabled and self._is_mixed_precision_supported()
        self.memory_efficient = memory_efficient

        # Determine optimal dtype
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self._select_optimal_dtype()

        # Initialize gradient scaler for CUDA
        self.scaler = GradScaler() if self.device == "cuda" and self.enabled else None

        logger.info(
            f"MixedPrecisionManager initialized: device={self.device}, "
            f"enabled={self.enabled}, dtype={self.dtype}"
        )

    def _select_device(self, device: str) -> str:
        """Select the best available device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _is_mixed_precision_supported(self) -> bool:
        """Check if mixed precision is supported on the current device."""
        if self.device == "cuda":
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif self.device == "mps":
            # MPS supports float16 but not bfloat16
            return torch.backends.mps.is_available()
        else:
            # CPU doesn't benefit from mixed precision
            return False

    def _select_optimal_dtype(self) -> torch.dtype:
        """Select optimal dtype based on device capabilities."""
        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        elif self.device == "mps":
            return torch.float16
        else:
            return torch.float32

    def get_autocast_context(self):
        """Get autocast context for mixed precision operations."""
        if not self.enabled:
            return _NoopContext()

        if self.device == "cuda":
            # Use torch.amp.autocast for CUDA
            try:
                return torch.amp.autocast("cuda", dtype=self.dtype)
            except Exception as e:
                logger.warning(f"CUDA autocast failed, using no-op: {e}")
                return _NoopContext()
        elif self.device == "mps":
            # Use torch.amp.autocast for MPS
            try:
                return torch.amp.autocast("mps", dtype=self.dtype)
            except Exception as e:
                logger.warning(f"MPS autocast not available, using no-op: {e}")
                return _NoopContext()
        else:
            # Keep fp32 on CPU
            return _NoopContext()

    def _mps_autocast_context(self):
        """Custom autocast context for MPS (manual dtype conversion)."""

        class MPSAutocastContext:
            def __init__(self, target_dtype):
                self.target_dtype = target_dtype
                self.original_dtypes = {}

            def __enter__(self):
                # Store original dtypes of model parameters
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return MPSAutocastContext(self.dtype)

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for mixed precision inference.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        if not self.enabled:
            return model

        # Convert model to appropriate dtype
        if self.dtype != torch.float32:
            model = model.to(dtype=self.dtype)
            logger.info(f"Converted model to {self.dtype}")

        # Enable memory efficient attention if available
        if self.memory_efficient and hasattr(model, "config"):
            self._enable_memory_efficient_attention(model)

        return model

    def _enable_memory_efficient_attention(self, model: nn.Module) -> None:
        """Enable memory efficient attention if supported."""
        try:
            if hasattr(model, "config"):
                # Set attention implementation to memory efficient
                if hasattr(model.config, "attention_implementation"):
                    model.config.attention_implementation = "flash_attention_2"
                    logger.info("Enabled memory efficient attention")
        except Exception as e:
            logger.warning(f"Failed to enable memory efficient attention: {e}")

    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor for mixed precision operations.

        Args:
            tensor: Input tensor

        Returns:
            Optimized tensor
        """
        if not self.enabled or tensor.dtype == self.dtype:
            return tensor

        # Convert to optimal dtype
        return tensor.to(dtype=self.dtype)

    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about current optimization settings."""
        return {
            "device": self.device,
            "enabled": self.enabled,
            "dtype": str(self.dtype),
            "memory_efficient": self.memory_efficient,
            "gradient_scaler": self.scaler is not None,
        }


class GradientCheckpointingManager:
    """Manages gradient checkpointing for memory optimization."""

    def __init__(
        self,
        enabled: bool = True,
        checkpoint_ratio: float = 1.0,
        use_reentrant: bool = True,
    ):
        """
        Initialize gradient checkpointing manager.

        Args:
            enabled: Whether gradient checkpointing is enabled
            checkpoint_ratio: Fraction of layers to checkpoint (0.0 to 1.0)
            use_reentrant: Whether to use reentrant checkpointing
        """
        self.enabled = enabled
        self.checkpoint_ratio = max(0.0, min(1.0, checkpoint_ratio))
        self.use_reentrant = use_reentrant

        logger.info(
            f"GradientCheckpointingManager initialized: enabled={enabled}, "
            f"checkpoint_ratio={checkpoint_ratio}, use_reentrant={use_reentrant}"
        )

    def enable_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Enable gradient checkpointing on model.

        Args:
            model: PyTorch model to optimize

        Returns:
            Model with gradient checkpointing enabled
        """
        if not self.enabled:
            return model

        try:
            # Enable gradient checkpointing
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()  # type: ignore
                logger.info("Enabled gradient checkpointing")
            else:
                # Manual checkpointing for custom models
                self._apply_manual_checkpointing(model)

        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

        return model

    def _apply_manual_checkpointing(self, model: nn.Module) -> None:
        """Apply manual gradient checkpointing to model layers."""

        def checkpoint_wrapper(layer):
            if isinstance(
                layer, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
            ):
                return torch.utils.checkpoint.checkpoint(
                    layer, use_reentrant=self.use_reentrant
                )
            return layer

        # Apply to transformer layers - more robust approach
        try:
            for name, module in model.named_modules():
                if isinstance(
                    module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
                ):
                    # Try to replace with checkpointed version
                    try:
                        parent = model
                        attrs = name.split(".")
                        for attr in attrs[:-1]:
                            parent = getattr(parent, attr)

                        # Check if we can set the attribute
                        if hasattr(parent, attrs[-1]):
                            setattr(parent, attrs[-1], checkpoint_wrapper(module))
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Could not apply checkpointing to {name}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Manual checkpointing failed: {e}")

    def get_memory_savings_estimate(self, model: nn.Module) -> Dict[str, Any]:
        """
        Estimate memory savings from gradient checkpointing.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with memory savings estimates
        """
        if not self.enabled:
            return {"enabled": False, "estimated_savings": 0.0}

        # Count checkpointable layers
        checkpointable_layers = 0
        total_layers = 0

        for module in model.modules():
            if isinstance(
                module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
            ):
                total_layers += 1
                if torch.rand(1).item() < self.checkpoint_ratio:
                    checkpointable_layers += 1

        # Estimate memory savings (rough approximation)
        estimated_savings = (
            checkpointable_layers / max(total_layers, 1)
        ) * 0.5  # 50% savings per layer

        return {
            "enabled": True,
            "checkpointable_layers": checkpointable_layers,
            "total_layers": total_layers,
            "checkpoint_ratio": self.checkpoint_ratio,
            "estimated_savings": estimated_savings,
        }


class LocalOptimizationManager:
    """Combined optimization manager for local CPU/MPS development."""

    def __init__(
        self,
        device: str = "auto",
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
        memory_efficient: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize local optimization manager.

        Args:
            device: Device to optimize for
            mixed_precision: Whether to enable mixed precision
            gradient_checkpointing: Whether to enable gradient checkpointing
            memory_efficient: Whether to use memory efficient operations
            dtype: Specific dtype to use
        """
        self.device = device

        # Initialize sub-managers
        self.mixed_precision = MixedPrecisionManager(
            device=device,
            enabled=mixed_precision,
            dtype=dtype,
            memory_efficient=memory_efficient,
        )

        self.gradient_checkpointing = GradientCheckpointingManager(
            enabled=gradient_checkpointing,
        )

        logger.info(
            f"LocalOptimizationManager initialized: device={device}, "
            f"mixed_precision={mixed_precision}, "
            f"gradient_checkpointing={gradient_checkpointing}"
        )

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply all optimizations to a model.

        Args:
            model: PyTorch model to optimize

        Returns:
            Fully optimized model
        """
        # Apply mixed precision optimization
        model = self.mixed_precision.optimize_model(model)

        # Apply gradient checkpointing
        model = self.gradient_checkpointing.enable_checkpointing(model)

        logger.info("Applied all optimizations to model")
        return model

    def get_optimization_context(self):
        """Get optimization context for inference."""
        return self.mixed_precision.get_autocast_context()

    def get_optimization_info(self) -> Dict[str, Any]:
        """Get optimization information."""
        return self.mixed_precision.get_optimization_info()

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "device": self.device,
            "mixed_precision": self.mixed_precision.get_optimization_info(),
            "gradient_checkpointing": (
                self.gradient_checkpointing.get_memory_savings_estimate(
                    torch.nn.Linear(10, 10)  # Dummy model for estimation
                )
            ),
        }


def select_device_dtype(device: str = "auto") -> tuple[str, torch.dtype, bool]:
    """
    Select optimal device and dtype for the given device preference.

    Args:
        device: Device preference ("auto", "cpu", "mps", "cuda")

    Returns:
        Tuple of (selected_device, optimal_dtype, amp_enabled)
    """
    # Select device
    if device == "auto":
        if torch.backends.mps.is_available():
            selected_device = "mps"
        elif torch.cuda.is_available():
            selected_device = "cuda"
        else:
            selected_device = "cpu"
    else:
        selected_device = device

    # Check environment overrides
    env_amp = os.getenv("SPECDEC_AMP")
    amp_enabled = True
    if env_amp is not None:
        amp_enabled = env_amp.lower() in ("1", "true", "yes")

    env_dtype = os.getenv("SPECDEC_DTYPE")
    if env_dtype is not None:
        if env_dtype.lower() == "float16":
            optimal_dtype = torch.float16
        elif env_dtype.lower() == "bfloat16":
            optimal_dtype = torch.bfloat16
        elif env_dtype.lower() == "float32":
            optimal_dtype = torch.float32
        else:
            optimal_dtype = _get_default_dtype(selected_device)
    else:
        optimal_dtype = _get_default_dtype(selected_device)

    # Disable AMP for CPU
    if selected_device == "cpu":
        amp_enabled = False
        optimal_dtype = torch.float32

    return selected_device, optimal_dtype, amp_enabled


def _get_default_dtype(device: str) -> torch.dtype:
    """Get default dtype for device."""
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device == "mps":
        return torch.float16
    else:
        return torch.float32


def amp_context(device: str, dtype: torch.dtype) -> Any:
    """
    Get appropriate autocast context for device and dtype.

    Args:
        device: Target device ("cuda", "mps", "cpu")
        dtype: Target dtype

    Returns:
        Autocast context manager
    """
    if device == "cuda":
        try:
            return torch.amp.autocast("cuda", dtype=dtype)
        except Exception as e:
            logger.warning(f"CUDA autocast failed, using no-op: {e}")
            return _NoopContext()
    elif device == "mps":
        try:
            return torch.amp.autocast("mps", dtype=dtype)
        except Exception as e:
            logger.warning(f"MPS autocast failed, using no-op: {e}")
            return _NoopContext()
    else:
        return _NoopContext()


def create_optimization_manager(
    device: str = "auto",
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    memory_efficient: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> LocalOptimizationManager:
    """
    Create a LocalOptimizationManager instance.

    Args:
        device: Device to optimize for
        mixed_precision: Whether to enable mixed precision
        gradient_checkpointing: Whether to enable gradient checkpointing
        memory_efficient: Whether to use memory efficient operations
        dtype: Specific dtype to use

    Returns:
        Configured LocalOptimizationManager instance
    """
    return LocalOptimizationManager(
        device=device,
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        memory_efficient=memory_efficient,
        dtype=dtype,
    )
