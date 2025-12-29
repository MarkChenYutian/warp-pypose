# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_VEC4,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec4,
    wp_quat,
)


# =============================================================================
# Helper to extract first 3 components from vec4 as vec3
# =============================================================================

def _make_extract_vec3(dtype):
    vec3_ctor = DTYPE_TO_VEC3[dtype]

    @wp.func
    def extract_vec3(v: T.Any) -> T.Any:
        return vec3_ctor(v[0], v[1], v[2])
    return extract_vec3


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
#
# From PyPose SO3_Act4.backward:
#   X_grad_xyz = out[:3] × grad[:3]
#   p_grad[:3] = R^T @ grad[:3]
#   p_grad[3] = grad[3]
#
# OPTIMIZATION: Uses wp.quat_rotate_inv(q, g3) instead of transpose(quat_to_matrix(q)) @ g3.
# This is mathematically equivalent but more efficient.
# =============================================================================

def _make_kernel_1d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        q = X[i]
        o = out[i]
        g = grad_output[i]
        
        o3 = extract_vec3(o)
        g3 = extract_vec3(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.quat_rotate_inv(q, g3)
        grad_p[i] = vec4_ctor(pg3[0], pg3[1], pg3[2], g[3])
    return implement


def _make_kernel_2d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        q = X[i, j]
        o = out[i, j]
        g = grad_output[i, j]
        
        o3 = extract_vec3(o)
        g3 = extract_vec3(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i, j] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.quat_rotate_inv(q, g3)
        grad_p[i, j] = vec4_ctor(pg3[0], pg3[1], pg3[2], g[3])
    return implement


def _make_kernel_3d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        q = X[i, j, k]
        o = out[i, j, k]
        g = grad_output[i, j, k]
        
        o3 = extract_vec3(o)
        g3 = extract_vec3(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i, j, k] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.quat_rotate_inv(q, g3)
        grad_p[i, j, k] = vec4_ctor(pg3[0], pg3[1], pg3[2], g[3])
    return implement


def _make_kernel_4d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        q = X[i, j, k, l]
        o = out[i, j, k, l]
        g = grad_output[i, j, k, l]
        
        o3 = extract_vec3(o)
        g3 = extract_vec3(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i, j, k, l] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.quat_rotate_inv(q, g3)
        grad_p[i, j, k, l] = vec4_ctor(pg3[0], pg3[1], pg3[2], g[3])
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

def SO3_Act4_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_Act4 with respect to inputs.
    
    Args:
        X: Input quaternion of shape (..., 4)
        out: Forward pass output of shape (..., 4)
        grad_output: Gradient w.r.t output of shape (..., 4)
        
    Returns:
        Tuple of (grad_X, grad_p):
            grad_X: shape (..., 4) with w component = 0
            grad_p: shape (..., 4)
    """
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    
    quat_type = wp_quat(dtype)
    vec4_type = wp_vec4(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    X_wp = wp.from_torch(X, dtype=quat_type)
    out_wp = wp.from_torch(out, dtype=vec4_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec4_type)
    
    grad_X_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec4_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info), finalize_output(grad_p_tensor, batch_info)
