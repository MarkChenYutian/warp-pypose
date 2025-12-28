# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
)


# =============================================================================
# SE3_Mul Forward Pass
#
# SE3 multiplication (composition):
#   t_out = t_X + R_X @ t_Y
#   q_out = q_X * q_Y
# =============================================================================


def _make_se3_mul(dtype):
    @wp.func
    def se3_mul(X: T.Any, Y: T.Any) -> T.Any:
        """Compose two SE3 transforms: out = X @ Y"""
        return wp.transform_multiply(X, Y)
    return se3_mul


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        Y: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_mul(X[i], Y[i])
    return implement


def _make_kernel_2d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        Y: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_mul(X[i, j], Y[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        Y: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_mul(X[i, j, k], Y[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        Y: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_mul(X[i, j, k, l], Y[i, j, k, l])
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
}


# =============================================================================
# Main forward function
# =============================================================================

def SE3_Mul_fwd(X: pp.LieTensor, Y: pp.LieTensor) -> torch.Tensor:
    """
    Compose two SE3 transformations: out = X @ Y.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SE3 LieTensor of shape (..., 7)
        Y: SE3 LieTensor of shape (..., 7)
        
    Returns:
        Composed SE3 tensor of shape (broadcast(...), 7)
    """
    X_tensor = X.tensor()
    Y_tensor = Y.tensor() if hasattr(Y, 'tensor') else Y
    
    # Prepare batch dimensions with broadcasting
    X_tensor, Y_tensor, batch_info = prepare_batch_broadcast(X_tensor, Y_tensor)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 7).contiguous()
    Y_expanded = Y_tensor.expand(*batch_info.shape, 7).contiguous()
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=transform_type)
    Y_wp = wp.from_torch(Y_expanded, dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=transform_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, Y_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
