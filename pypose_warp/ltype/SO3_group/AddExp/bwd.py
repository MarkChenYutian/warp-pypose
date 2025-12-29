# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for SO3 AddExp: fused Exp(delta) * X operation.

Given Y = Exp(delta) * X, we need gradients w.r.t. delta and X.

The chain rule gives us:
- grad_delta = grad_Y[..., :3] @ Jl(delta)
- grad_X = (grad_Y[..., :3] @ R, 0) where R = quat_to_matrix(Exp(delta))

This is derived from:
1. Q = Exp(delta): d(Q)/d(delta) uses left Jacobian Jl
2. Y = Q * X: d(Y)/d(Q) = (grad[..., :3], 0), d(Y)/d(X) = (grad[..., :3] @ R_Q, 0)

OPTIMIZATION: Fused computation that computes theta=length(delta) and trig values
only once for both grad_delta and grad_X calculations.
"""

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_QUAT,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    get_eps_for_dtype,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# Uses fused computation to share theta and trig values between gradients.
# =============================================================================

def _make_kernel_1d(dtype):
    """Fused 1D kernel for AddExp backward."""
    # Get epsilon values for the different divisions
    eps_jl = get_eps_for_dtype(dtype, power=3)  # For Jl (theta^3 division)
    eps_exp = get_eps_for_dtype(dtype, power=2)  # For exp (theta division)
    
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_delta: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        d = delta[i]
        g = grad_output[i]
        grad_xyz = vec3_ctor(g[0], g[1], g[2])
        
        # =====================================================================
        # SHARED COMPUTATION: theta and derived values (computed once)
        # =====================================================================
        theta = wp.length(d)
        theta2 = theta * theta
        K = wp.skew(d)
        I = wp.identity(n=3, dtype=dtype)
        
        # =====================================================================
        # GRAD_DELTA: Uses Jl(delta)
        # Jl = I + coef1 * K + coef2 * (K @ K)
        # =====================================================================
        eps_jl_val = dtype(eps_jl)
        coef1_jl = dtype(0.0)
        coef2_jl = dtype(0.0)
        
        if theta > eps_jl_val:
            coef1_jl = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2_jl = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            coef1_jl = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2_jl = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        
        Jl = I + coef1_jl * K + coef2_jl * (K @ K)
        grad_delta[i] = wp.transpose(Jl) @ grad_xyz
        
        # =====================================================================
        # GRAD_X: Uses Exp(delta) then quat_to_matrix
        # Q = Exp(delta), R = quat_to_matrix(Q), grad_X = R^T @ grad_xyz
        # =====================================================================
        theta4 = theta2 * theta2
        theta_half = dtype(0.5) * theta
        eps_exp_val = dtype(eps_exp)
        
        imag_factor = dtype(0.0)
        real_factor = dtype(0.0)
        
        if theta > eps_exp_val:
            imag_factor = wp.sin(theta_half) / theta
            real_factor = wp.cos(theta_half)
        else:
            imag_factor = dtype(0.5) - (dtype(1.0) / dtype(48.0)) * theta2 + (dtype(1.0) / dtype(3840.0)) * theta4
            real_factor = dtype(1.0) - (dtype(1.0) / dtype(8.0)) * theta2 + (dtype(1.0) / dtype(384.0)) * theta4
        
        Q = quat_ctor(d[0] * imag_factor, d[1] * imag_factor, d[2] * imag_factor, real_factor)
        R = wp.quat_to_matrix(Q)
        grad_vec = wp.transpose(R) @ grad_xyz
        grad_X[i] = quat_ctor(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    
    return implement


def _make_kernel_2d(dtype):
    """Fused 2D kernel for AddExp backward."""
    eps_jl = get_eps_for_dtype(dtype, power=3)
    eps_exp = get_eps_for_dtype(dtype, power=2)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_delta: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        d = delta[i, j]
        g = grad_output[i, j]
        grad_xyz = vec3_ctor(g[0], g[1], g[2])
        
        # SHARED
        theta = wp.length(d)
        theta2 = theta * theta
        K = wp.skew(d)
        I = wp.identity(n=3, dtype=dtype)
        
        # GRAD_DELTA
        eps_jl_val = dtype(eps_jl)
        coef1_jl = dtype(0.0)
        coef2_jl = dtype(0.0)
        if theta > eps_jl_val:
            coef1_jl = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2_jl = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            coef1_jl = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2_jl = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        Jl = I + coef1_jl * K + coef2_jl * (K @ K)
        grad_delta[i, j] = wp.transpose(Jl) @ grad_xyz
        
        # GRAD_X
        theta4 = theta2 * theta2
        theta_half = dtype(0.5) * theta
        eps_exp_val = dtype(eps_exp)
        imag_factor = dtype(0.0)
        real_factor = dtype(0.0)
        if theta > eps_exp_val:
            imag_factor = wp.sin(theta_half) / theta
            real_factor = wp.cos(theta_half)
        else:
            imag_factor = dtype(0.5) - (dtype(1.0) / dtype(48.0)) * theta2 + (dtype(1.0) / dtype(3840.0)) * theta4
            real_factor = dtype(1.0) - (dtype(1.0) / dtype(8.0)) * theta2 + (dtype(1.0) / dtype(384.0)) * theta4
        Q = quat_ctor(d[0] * imag_factor, d[1] * imag_factor, d[2] * imag_factor, real_factor)
        R = wp.quat_to_matrix(Q)
        grad_vec = wp.transpose(R) @ grad_xyz
        grad_X[i, j] = quat_ctor(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    
    return implement


def _make_kernel_3d(dtype):
    """Fused 3D kernel for AddExp backward."""
    eps_jl = get_eps_for_dtype(dtype, power=3)
    eps_exp = get_eps_for_dtype(dtype, power=2)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_delta: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        d = delta[i, j, k]
        g = grad_output[i, j, k]
        grad_xyz = vec3_ctor(g[0], g[1], g[2])
        
        # SHARED
        theta = wp.length(d)
        theta2 = theta * theta
        K = wp.skew(d)
        I = wp.identity(n=3, dtype=dtype)
        
        # GRAD_DELTA
        eps_jl_val = dtype(eps_jl)
        coef1_jl = dtype(0.0)
        coef2_jl = dtype(0.0)
        if theta > eps_jl_val:
            coef1_jl = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2_jl = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            coef1_jl = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2_jl = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        Jl = I + coef1_jl * K + coef2_jl * (K @ K)
        grad_delta[i, j, k] = wp.transpose(Jl) @ grad_xyz
        
        # GRAD_X
        theta4 = theta2 * theta2
        theta_half = dtype(0.5) * theta
        eps_exp_val = dtype(eps_exp)
        imag_factor = dtype(0.0)
        real_factor = dtype(0.0)
        if theta > eps_exp_val:
            imag_factor = wp.sin(theta_half) / theta
            real_factor = wp.cos(theta_half)
        else:
            imag_factor = dtype(0.5) - (dtype(1.0) / dtype(48.0)) * theta2 + (dtype(1.0) / dtype(3840.0)) * theta4
            real_factor = dtype(1.0) - (dtype(1.0) / dtype(8.0)) * theta2 + (dtype(1.0) / dtype(384.0)) * theta4
        Q = quat_ctor(d[0] * imag_factor, d[1] * imag_factor, d[2] * imag_factor, real_factor)
        R = wp.quat_to_matrix(Q)
        grad_vec = wp.transpose(R) @ grad_xyz
        grad_X[i, j, k] = quat_ctor(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    
    return implement


def _make_kernel_4d(dtype):
    """Fused 4D kernel for AddExp backward."""
    eps_jl = get_eps_for_dtype(dtype, power=3)
    eps_exp = get_eps_for_dtype(dtype, power=2)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_delta: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        d = delta[i, j, k, l]
        g = grad_output[i, j, k, l]
        grad_xyz = vec3_ctor(g[0], g[1], g[2])
        
        # SHARED
        theta = wp.length(d)
        theta2 = theta * theta
        K = wp.skew(d)
        I = wp.identity(n=3, dtype=dtype)
        
        # GRAD_DELTA
        eps_jl_val = dtype(eps_jl)
        coef1_jl = dtype(0.0)
        coef2_jl = dtype(0.0)
        if theta > eps_jl_val:
            coef1_jl = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2_jl = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            coef1_jl = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2_jl = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        Jl = I + coef1_jl * K + coef2_jl * (K @ K)
        grad_delta[i, j, k, l] = wp.transpose(Jl) @ grad_xyz
        
        # GRAD_X
        theta4 = theta2 * theta2
        theta_half = dtype(0.5) * theta
        eps_exp_val = dtype(eps_exp)
        imag_factor = dtype(0.0)
        real_factor = dtype(0.0)
        if theta > eps_exp_val:
            imag_factor = wp.sin(theta_half) / theta
            real_factor = wp.cos(theta_half)
        else:
            imag_factor = dtype(0.5) - (dtype(1.0) / dtype(48.0)) * theta2 + (dtype(1.0) / dtype(3840.0)) * theta4
            real_factor = dtype(1.0) - (dtype(1.0) / dtype(8.0)) * theta2 + (dtype(1.0) / dtype(384.0)) * theta4
        Q = quat_ctor(d[0] * imag_factor, d[1] * imag_factor, d[2] * imag_factor, real_factor)
        R = wp.quat_to_matrix(Q)
        grad_vec = wp.transpose(R) @ grad_xyz
        grad_X[i, j, k, l] = quat_ctor(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    
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

def SO3_AddExp_bwd(
    delta: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_AddExp with respect to inputs.
    
    Given Y = Exp(delta) * X, computes:
    - grad_delta: Gradient w.r.t. delta (so3 axis-angle)
    - grad_X: Gradient w.r.t. X (SO3 quaternion)
    
    Args:
        delta: Forward input (so3 axis-angle) of shape (..., 3)
        grad_output: Gradient w.r.t. output quaternion of shape (..., 4)
        
    Returns:
        Tuple of (grad_delta, grad_X):
        - grad_delta: shape (..., 3)
        - grad_X: shape (..., 4)
    """
    # Prepare batch dimensions
    delta, batch_info = prepare_batch_single(delta)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = delta.dtype
    device = delta.device
    
    vec3_type = wp_vec3(dtype)
    quat_type = wp_quat(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    delta = delta.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    delta_wp = wp.from_torch(delta, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=quat_type)
    
    # Create output tensors
    grad_delta_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=device)
    grad_X_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    
    grad_delta_wp = wp.from_torch(grad_delta_tensor, dtype=vec3_type)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=delta_wp.device,
        inputs=[delta_wp, grad_output_wp, grad_delta_wp, grad_X_wp],
    )
    
    return finalize_output(grad_delta_tensor, batch_info), finalize_output(grad_X_tensor, batch_info)
