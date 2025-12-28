# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# Warp requires fixed ndim at compile time, so we need separate kernels.
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = wp.quat_rotate(X[i], p[i])
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = wp.quat_rotate(X[i, j], p[i, j])
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = wp.quat_rotate(X[i, j, k], p[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = wp.quat_rotate(X[i, j, k, l], p[i, j, k, l])
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

def SO3_Act_fwd(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
    """
    Apply SO3 rotation X to 3D points p.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        p: Tensor of shape (..., 3) - 3D points
        
    Returns:
        Rotated points of shape (broadcast(...), 3)
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions with broadcasting
    X_tensor, p, batch_info = prepare_batch_broadcast(X_tensor, p)
    
    # Expand tensors to broadcast shape (creates stride-0 views, no data copy)
    X_expanded = X_tensor.expand(*batch_info.shape, 4)
    p_expanded = p.expand(*batch_info.shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
