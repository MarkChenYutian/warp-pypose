# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type
from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        Y: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = wp.mul(X[i], Y[i])
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        Y: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = wp.mul(X[i, j], Y[i, j])
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        Y: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = wp.mul(X[i, j, k], Y[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        Y: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = wp.mul(X[i, j, k, l], Y[i, j, k, l])
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

def SO3_Mul_fwd(X: pp.LieTensor, Y: pp.LieTensor) -> pp.LieTensor:
    """
    Compute SO3 group multiplication X * Y (quaternion multiplication).
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        Y: SO3 LieTensor of shape (..., 4) - quaternion representation
        
    Returns:
        SO3 LieTensor of shape (broadcast(...), 4) - product quaternion
    """
    X_tensor = X.tensor()
    Y_tensor = Y.tensor()
    
    # Prepare batch dimensions with broadcasting
    X_tensor, Y_tensor, batch_info = prepare_batch_broadcast(X_tensor, Y_tensor)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 4)
    Y_expanded = Y_tensor.expand(*batch_info.shape, 4)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    Y_wp = wp.from_torch(Y_expanded, dtype=quat_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=quat_type)
    
    # Get kernel and launch (dtype is ignored for wp.mul)
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, None)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, Y_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    from ... import warpSO3_type  # lazy import to avoid circular import
    return pp.LieTensor(out_tensor, ltype=warpSO3_type)
