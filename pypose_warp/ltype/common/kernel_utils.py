# pyright: reportInvalidTypeForm=false
"""
Centralized utilities for Warp kernel management.

This module provides:
- Type mappings between PyTorch and Warp types
- Kernel caching infrastructure
- Common batch handling utilities
"""

import torch
import warp as wp
import typing as T
from typing import Any, Callable


# =============================================================================
# PyTorch to Warp Type Mappings
# =============================================================================

TORCH_TO_WP_SCALAR: dict[torch.dtype, type] = {
    torch.float16: wp.float16,
    torch.float32: wp.float32,
    torch.float64: wp.float64,
}


# =============================================================================
# Warp Scalar to Vector/Matrix Type Mappings
# =============================================================================

DTYPE_TO_VEC3: dict[type, type] = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}

DTYPE_TO_VEC4: dict[type, type] = {
    wp.float16: wp.vec4h,
    wp.float32: wp.vec4f,
    wp.float64: wp.vec4d,
}

DTYPE_TO_QUAT: dict[type, type] = {
    wp.float16: wp.quath,
    wp.float32: wp.quatf,
    wp.float64: wp.quatd,
}

DTYPE_TO_MAT33: dict[type, type] = {
    wp.float16: wp.mat33h,
    wp.float32: wp.mat33f,
    wp.float64: wp.mat33d,
}

DTYPE_TO_TRANSFORM: dict[type, type] = {
    wp.float16: wp.transformh,
    wp.float32: wp.transformf,
    wp.float64: wp.transformd,
}


# =============================================================================
# Numerical Stability - Dtype-specific epsilon thresholds
# =============================================================================

# FP16 has limited precision: min positive = 6.1e-5, epsilon = 9.77e-4
# When dividing by theta^n, we need theta^n > min_positive to avoid underflow
# These thresholds are derived from (6.1e-5)^(1/n) with safety margin

_FP16_EPS_BY_POWER: dict[int, float] = {
    2: 0.02,    # theta^2 underflows when theta < 0.008, extra margin for stability
    3: 0.08,    # theta^3 underflows when theta < 0.04, extra margin for stability
    4: 0.12,    # theta^4 underflows when theta < 0.09, extra margin for stability
    5: 0.20,    # theta^5 underflows when theta < 0.14, extra margin for stability
}

_FP32_EPS = 1e-6
_FP64_EPS = 1e-12


def get_eps_for_dtype(dtype, power: int = 2) -> float:
    """
    Get appropriate epsilon threshold for divisions by theta^power.
    
    This is critical for numerical stability in FP16 where theta^n can
    underflow to zero, causing division by zero and NaN results.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        power: The power of theta in the denominator (2, 3, 4, or 5)
        
    Returns:
        Appropriate epsilon threshold for the given dtype and power
        
    Example:
        For so3_Jl_inv which divides by theta^2:
            eps = get_eps_for_dtype(wp.float16, power=2)  # returns 0.02
            
        For calcQ which divides by theta^5:
            eps = get_eps_for_dtype(wp.float16, power=5)  # returns 0.20
    """
    if dtype == wp.float16:
        return _FP16_EPS_BY_POWER.get(power, 0.05)
    elif dtype == wp.float32:
        return _FP32_EPS
    else:
        return _FP64_EPS


def get_eps_for_torch_dtype(dtype: torch.dtype, power: int = 2) -> float:
    """
    Get appropriate epsilon threshold for PyTorch tensors.
    
    This mirrors get_eps_for_dtype but works with PyTorch dtypes instead of
    Warp dtypes. Useful for backward passes that use PyTorch autograd.
    
    Args:
        dtype: PyTorch dtype (torch.float16, torch.float32, torch.float64)
        power: The power of theta in the denominator (2, 3, 4, or 5)
        
    Returns:
        Appropriate epsilon threshold for the given dtype and power
        
    Example:
        For so3_Jr_bwd which divides by theta^3:
            eps = get_eps_for_torch_dtype(torch.float16, power=3)  # returns 0.08
    """
    if dtype == torch.float16:
        return _FP16_EPS_BY_POWER.get(power, 0.05)
    elif dtype == torch.float32:
        return _FP32_EPS
    else:
        return _FP64_EPS


# =============================================================================
# Kernel Registry - Global cache for instantiated kernels
# =============================================================================

class KernelRegistry:
    """
    Global registry for caching instantiated Warp kernels.
    
    Kernels are cached by (factory_id, ndim, dtype) to avoid redundant
    compilation and ensure efficient kernel reuse.
    """
    
    _cache: dict[tuple[int, int, type], Any] = {}
    
    @classmethod
    def get(
        cls,
        factories: dict[int, Callable],
        ndim: int,
        dtype: type,
    ) -> Any:
        """
        Get or create a kernel for the given factory dict, ndim, and dtype.
        
        Args:
            factories: Dict mapping ndim -> kernel factory function
            ndim: Number of batch dimensions (1-4)
            dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
            
        Returns:
            Instantiated Warp kernel
        """
        # Use id of the factories dict as part of the key to distinguish
        # different kernel families
        key = (id(factories), ndim, dtype)
        if key not in cls._cache:
            factory = factories[ndim]
            cls._cache[key] = factory(dtype)
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        """Clear the kernel cache (useful for testing)."""
        cls._cache.clear()


# =============================================================================
# Batch Handling Utilities
# =============================================================================

class BatchInfo:
    """Container for batch dimension information."""
    __slots__ = ('shape', 'ndim', 'squeeze_output')
    
    def __init__(self, shape: tuple, ndim: int, squeeze_output: bool):
        self.shape = shape
        self.ndim = ndim
        self.squeeze_output = squeeze_output


def prepare_batch_single(
    tensor: torch.Tensor,
    max_ndim: int = 4,
) -> tuple[torch.Tensor, BatchInfo]:
    """
    Prepare a single tensor for batch processing.
    
    Handles scalar case (adds dummy dimension) and validates ndim.
    
    Args:
        tensor: Input tensor of shape (..., D) where D is the feature dim
        max_ndim: Maximum supported batch dimensions
        
    Returns:
        Tuple of (prepared_tensor, BatchInfo)
    """
    batch_shape = tensor.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        tensor = tensor.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > max_ndim:
        raise NotImplementedError(
            f"Batch dimensions > {max_ndim} not supported. Got shape {batch_shape}"
        )
    
    return tensor, BatchInfo(batch_shape, ndim, squeeze_output)


def prepare_batch_broadcast(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    max_ndim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, BatchInfo]:
    """
    Prepare two tensors for broadcast batch processing.
    
    Computes broadcast shape, handles scalar case, and validates ndim.
    
    Args:
        tensor1: First input tensor of shape (..., D1)
        tensor2: Second input tensor of shape (..., D2)
        max_ndim: Maximum supported batch dimensions
        
    Returns:
        Tuple of (prepared_tensor1, prepared_tensor2, BatchInfo)
    """
    batch_shape1 = tensor1.shape[:-1]
    batch_shape2 = tensor2.shape[:-1]
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(batch_shape1, batch_shape2)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {tensor1.shape} and {tensor2.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        tensor1 = tensor1.unsqueeze(0)
        tensor2 = tensor2.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > max_ndim:
        raise NotImplementedError(
            f"Batch dimensions > {max_ndim} not supported. Got shape {out_batch_shape}"
        )
    
    return tensor1, tensor2, BatchInfo(out_batch_shape, ndim, squeeze_output)


def finalize_output(tensor: torch.Tensor, batch_info: BatchInfo) -> torch.Tensor:
    """
    Finalize output tensor by squeezing if necessary.
    
    Args:
        tensor: Output tensor
        batch_info: BatchInfo from prepare_batch_*
        
    Returns:
        Finalized tensor (squeezed if input was scalar)
    """
    if batch_info.squeeze_output:
        return tensor.squeeze(0)
    return tensor


# =============================================================================
# Gradient Reduction Utilities
# =============================================================================

def compute_reduce_dims(
    input_shape: tuple,
    output_shape: tuple,
) -> list[int]:
    """
    Compute dimensions to reduce for gradient broadcasting.
    
    When computing gradients with broadcasting, we need to sum over
    dimensions where the input was broadcast (had size 1).
    
    Args:
        input_shape: Original batch shape of input
        output_shape: Broadcast output batch shape
        
    Returns:
        List of dimension indices to reduce
    """
    ndim_out = len(output_shape)
    ndim_in = len(input_shape)
    
    # Pad input shape to match output shape
    padded = (1,) * (ndim_out - ndim_in) + tuple(input_shape)
    
    # Find dims where input was broadcast
    reduce_dims = []
    for i, (in_dim, out_dim) in enumerate(zip(padded, output_shape)):
        if in_dim == 1 and out_dim > 1:
            reduce_dims.append(i)
    
    return reduce_dims


def reduce_gradient(
    grad: torch.Tensor,
    original_shape: tuple,
    output_shape: tuple,
    feature_dim: int,
) -> torch.Tensor:
    """
    Reduce gradient to original shape after broadcasting.
    
    Args:
        grad: Gradient tensor at broadcast shape
        original_shape: Original batch shape of the input
        output_shape: Broadcast output batch shape
        feature_dim: Size of the feature dimension
        
    Returns:
        Gradient reduced to original shape
    """
    reduce_dims = compute_reduce_dims(original_shape, output_shape)
    
    if reduce_dims:
        grad = grad.sum(dim=reduce_dims, keepdim=True)
    
    return grad.view(*original_shape, feature_dim)

