# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.warp_functions import SO3_log_wp_func, so3_Jl_inv
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Backward kernel for SO3_Jinvp
#
# Forward: out = Jl_inv(Log(X)) @ p
#
# Given grad_output (gradient w.r.t. output), compute:
#   - grad_X: gradient w.r.t. quaternion X (shape ..., 4)
#   - grad_p: gradient w.r.t. tangent vector p (shape ..., 3)
# =============================================================================

def _make_compute_jinvp_grad(dtype):
    """Generate dtype-specific backward computation for Jinvp."""
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    
    @wp.func
    def compute_grad_X(q: T.Any, p: T.Any, grad_out: T.Any) -> T.Any:
        """Compute gradient w.r.t. X (quaternion)."""
        so3 = SO3_log_wp_func(q)
        K = wp.skew(so3)
        I = wp.identity(n=3, dtype=dtype)
        theta = wp.length(so3)
        
        eps = dtype(1e-6)
        coef2 = dtype(0.0)
        dcoef2_dtheta = dtype(0.0)
        
        if theta > eps:
            theta_half = dtype(0.5) * theta
            theta2 = theta * theta
            sin_half = wp.sin(theta_half)
            cos_half = wp.cos(theta_half)
            cot_half = cos_half / sin_half
            
            coef2 = (dtype(1.0) - theta * cot_half * dtype(0.5)) / theta2
            
            sin2_half = sin_half * sin_half
            df_dtheta = -cot_half * dtype(0.5) + theta / (dtype(4.0) * sin2_half)
            f = dtype(1.0) - theta * cot_half * dtype(0.5)
            dcoef2_dtheta = (df_dtheta * theta2 - f * dtype(2.0) * theta) / (theta2 * theta2)
        else:
            coef2 = dtype(1.0) / dtype(12.0)
            dcoef2_dtheta = dtype(0.0)
        
        Jl_inv = I - dtype(0.5) * K + coef2 * (K @ K)
        K2 = K @ K
        
        # Skew derivative matrices
        dK0 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK0[1, 2] = dtype(-1.0)
        dK0[2, 1] = dtype(1.0)
        
        dK1 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK1[0, 2] = dtype(1.0)
        dK1[2, 0] = dtype(-1.0)
        
        dK2 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK2[0, 1] = dtype(-1.0)
        dK2[1, 0] = dtype(1.0)
        
        # Compute grad_so3[m]
        dcoef2_dso3_0 = dtype(0.0)
        if theta > eps:
            dcoef2_dso3_0 = dcoef2_dtheta * so3[0] / theta
        dJ0 = -dtype(0.5) * dK0 + dcoef2_dso3_0 * K2 + coef2 * (dK0 @ K + K @ dK0)
        grad_so3_0 = wp.dot(grad_out, dJ0 @ p)
        
        dcoef2_dso3_1 = dtype(0.0)
        if theta > eps:
            dcoef2_dso3_1 = dcoef2_dtheta * so3[1] / theta
        dJ1 = -dtype(0.5) * dK1 + dcoef2_dso3_1 * K2 + coef2 * (dK1 @ K + K @ dK1)
        grad_so3_1 = wp.dot(grad_out, dJ1 @ p)
        
        dcoef2_dso3_2 = dtype(0.0)
        if theta > eps:
            dcoef2_dso3_2 = dcoef2_dtheta * so3[2] / theta
        dJ2 = -dtype(0.5) * dK2 + dcoef2_dso3_2 * K2 + coef2 * (dK2 @ K + K @ dK2)
        grad_so3_2 = wp.dot(grad_out, dJ2 @ p)
        
        grad_so3 = wp.vector(grad_so3_0, grad_so3_1, grad_so3_2)
        
        # Backprop through Log: grad_X = [Jl_inv^T @ grad_so3, 0]
        grad_X_xyz = wp.transpose(Jl_inv) @ grad_so3
        grad_X = wp.quaternion(grad_X_xyz[0], grad_X_xyz[1], grad_X_xyz[2], dtype(0.0))
        
        return grad_X
    
    @wp.func
    def compute_grad_p(q: T.Any, grad_out: T.Any) -> T.Any:
        """Compute grad_p = Jl_inv^T @ grad_out."""
        so3 = SO3_log_wp_func(q)
        Jl_inv = so3_Jl_inv_impl(so3)
        return wp.transpose(Jl_inv) @ grad_out
    
    return compute_grad_X, compute_grad_p


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad_X, compute_grad_p = _make_compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad_X(X[i], p[i], grad_output[i])
        grad_p[i] = compute_grad_p(X[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad_X, compute_grad_p = _make_compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad_X(X[i, j], p[i, j], grad_output[i, j])
        grad_p[i, j] = compute_grad_p(X[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad_X, compute_grad_p = _make_compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad_X(X[i, j, k], p[i, j, k], grad_output[i, j, k])
        grad_p[i, j, k] = compute_grad_p(X[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad_X, compute_grad_p = _make_compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad_X(X[i, j, k, l], p[i, j, k, l], grad_output[i, j, k, l])
        grad_p[i, j, k, l] = compute_grad_p(X[i, j, k, l], grad_output[i, j, k, l])
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

def SO3_Jinvp_bwd(
    X: torch.Tensor,
    p: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SO3_Jinvp.
    
    Args:
        X: Quaternion tensor of shape (..., 4)
        p: Tangent vector tensor of shape (..., 3)
        grad_output: Gradient w.r.t. output, shape (..., 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 4)
        grad_p: Gradient w.r.t. p, shape (..., 3)
    """
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        p = p.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    p = p.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=quat_type)
    p_wp = wp.from_torch(p, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec3_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info), finalize_output(grad_p_tensor, batch_info)
