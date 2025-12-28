# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
#
# From PyPose SO3_AdjXa.backward:
#   X_grad = -grad_output @ so3_adj(out)  (where so3_adj = skew matrix)
#   a_grad = grad_output @ SO3_Adj(X)     (where SO3_Adj = rotation matrix)
#
# In PyPose these are row vector @ matrix. In Warp with column vectors:
#   X_grad_xyz = -grad_output @ skew(out) = -out × grad_output (cross product)
#   a_grad = grad_output @ R = R^T @ grad_output (for rotation back)
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_a: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        R = wp.quat_to_matrix(X[i])
        g = grad_output[i]
        o = out[i]
        
        grad_a[i] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_a: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j])
        g = grad_output[i, j]
        o = out[i, j]
        
        grad_a[i, j] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i, j] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_a: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k])
        g = grad_output[i, j, k]
        o = out[i, j, k]
        
        grad_a[i, j, k] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i, j, k] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_a: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k, l])
        g = grad_output[i, j, k, l]
        o = out[i, j, k, l]
        
        grad_a[i, j, k, l] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i, j, k, l] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
}


# =============================================================================
# Main backward function
# =============================================================================

def SO3_AdjXa_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_AdjXa with respect to inputs.
    
    Args:
        X: Input quaternion of shape (..., 4)
        out: Forward pass output of shape (..., 3)
        grad_output: Gradient w.r.t output of shape (..., 3)
        
    Returns:
        Tuple of (grad_X, grad_a):
            grad_X: shape (..., 4) with w component = 0
            grad_a: shape (..., 3)
    """
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    X_wp = wp.from_torch(X, dtype=quat_type)
    out_wp = wp.from_torch(out, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    grad_X_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    grad_a_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_a_wp = wp.from_torch(grad_a_tensor, dtype=vec3_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_a_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info), finalize_output(grad_a_tensor, batch_info)
