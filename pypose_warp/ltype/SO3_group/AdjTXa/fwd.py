# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
#
# SO3_AdjTXa forward:
#   out = SO3_Adj(X^{-1}) @ a = R^T @ a
# where R = quat_to_matrix(X) is the 3x3 rotation matrix.
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        a: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        R = wp.quat_to_matrix(X[i])
        out[i] = wp.transpose(R) @ a[i]
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        a: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j])
        out[i, j] = wp.transpose(R) @ a[i, j]
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        a: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k])
        out[i, j, k] = wp.transpose(R) @ a[i, j, k]
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        a: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k, l])
        out[i, j, k, l] = wp.transpose(R) @ a[i, j, k, l]
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

def SO3_AdjTXa_fwd(X: pp.LieTensor, a: torch.Tensor) -> torch.Tensor:
    """
    Compute the adjoint transpose action of SO3 element X on Lie algebra element a.
    
    This computes: out = SO3_Adj(X^{-1}) @ a = R^T @ a
    where R is the 3x3 rotation matrix corresponding to quaternion X.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        a: Tensor of shape (..., 3) - Lie algebra element (so3)
        
    Returns:
        Tensor of shape (broadcast(...), 3) - transformed Lie algebra element
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions with broadcasting
    X_tensor, a, batch_info = prepare_batch_broadcast(X_tensor, a)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 4)
    a_expanded = a.expand(*batch_info.shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    
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
    
    return finalize_output(out_tensor, batch_info)
