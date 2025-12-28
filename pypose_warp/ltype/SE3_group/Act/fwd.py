# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_vec3_type, wp_transform_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
)


# =============================================================================
# SE3_Act Forward Pass
#
# SE3 element X has shape (..., 7): [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# where t is translation (3D) and q is quaternion (4D)
#
# Action on point p: out = t + R @ p
# where R is the rotation matrix from quaternion q
# =============================================================================


def _make_se3_act_point(dtype):
    @wp.func
    def se3_act_point(X: T.Any, p: T.Any) -> T.Any:
        """Apply SE3 transform to a 3D point: t + R @ p"""
        return wp.transform_point(X, p)
    return se3_act_point


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_act_point = _make_se3_act_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_act_point(X[i], p[i])
    return implement


def _make_kernel_2d(dtype):
    se3_act_point = _make_se3_act_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_act_point(X[i, j], p[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_act_point = _make_se3_act_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_act_point(X[i, j, k], p[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_act_point = _make_se3_act_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_act_point(X[i, j, k, l], p[i, j, k, l])
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

def SE3_Act_fwd(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
    """
    Apply SE3 transformation X to 3D points p.
    
    SE3 action: out = t + R @ p
    where X = (t, q) with t being translation and q being rotation quaternion.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        p: Tensor of shape (..., 3) - 3D points
        
    Returns:
        Transformed points of shape (broadcast(...), 3)
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions with broadcasting
    X_tensor, p, batch_info = prepare_batch_broadcast(X_tensor, p)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 7)
    p_expanded = p.expand(*batch_info.shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=transform_type)
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
