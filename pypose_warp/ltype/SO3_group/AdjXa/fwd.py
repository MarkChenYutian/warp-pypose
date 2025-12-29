# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
#
# SO3_AdjXa forward:
#   out = SO3_Adj(X) @ a = R @ a
# where R = quat_to_matrix(X) is the 3x3 rotation matrix.
#
# OPTIMIZATION: Uses wp.quat_rotate(q, v) instead of quat_to_matrix(q) @ v.
# This is mathematically equivalent but more efficient as it avoids
# constructing the 3x3 rotation matrix.
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        a: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = wp.quat_rotate(X[i], a[i])
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        a: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = wp.quat_rotate(X[i, j], a[i, j])
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        a: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = wp.quat_rotate(X[i, j, k], a[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        a: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = wp.quat_rotate(X[i, j, k, l], a[i, j, k, l])
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

def SO3_AdjXa_fwd(X: pp.LieTensor, a: pp.LieTensor) -> pp.LieTensor:
    """
    Compute the adjoint action of SO3 element X on Lie algebra element a.
    
    This computes: out = SO3_Adj(X) @ a = R @ a
    where R is the 3x3 rotation matrix corresponding to quaternion X.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        a: Tensor of shape (..., 3) - Lie algebra element (so3)
        
    Returns:
        Tensor of shape (broadcast(...), 3) - rotated Lie algebra element
    """
    X_tensor = X.tensor()
    a_tensor = a.tensor()
    
    # Prepare batch dimensions with broadcasting
    X_tensor, a_tensor, batch_info = prepare_batch_broadcast(X_tensor, a_tensor)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 4)
    a_expanded = a_tensor.expand(*batch_info.shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat(dtype)
    vec3_type = wp_vec3(dtype)
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    a_wp = wp.from_torch(a_expanded, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, None)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, a_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    from ... import warpso3_type  # lazy import to avoid circular import
    return pp.LieTensor(out_tensor, ltype=warpso3_type)
