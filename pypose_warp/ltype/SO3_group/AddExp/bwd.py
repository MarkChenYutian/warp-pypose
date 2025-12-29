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
"""

import torch
import warp as wp
import typing as T

from ...common.warp_functions import so3_Jl, so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Helper function for computing AddExp backward
# =============================================================================

def _make_compute_addexp_grad(dtype):
    """Create gradient computation function for the given dtype."""
    so3_Jl_impl = so3_Jl(dtype)
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.func
    def compute_addexp_grad(delta: T.Any, grad_quat: T.Any) -> T.Any:
        """
        Compute grad_delta for AddExp backward.
        
        grad_delta = grad_Y[..., :3] @ Jl(delta)
        """
        Jl = so3_Jl_impl(delta)
        grad_xyz = wp.vector(grad_quat[0], grad_quat[1], grad_quat[2], dtype=dtype)
        return wp.transpose(Jl) @ grad_xyz
    
    @wp.func
    def compute_grad_X(delta: T.Any, grad_quat: T.Any) -> T.Any:
        """
        Compute grad_X for AddExp backward.
        
        grad_X = (grad_Y[..., :3] @ R, 0) where R = quat_to_matrix(Exp(delta))
        In warp column-vector form: grad_X_vec = R^T @ grad_Y_vec
        """
        Q = so3_exp_impl(delta)
        R = wp.quat_to_matrix(Q)
        grad_xyz = wp.vector(grad_quat[0], grad_quat[1], grad_quat[2], dtype=dtype)
        grad_vec = wp.transpose(R) @ grad_xyz
        return wp.quaternion(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    
    return compute_addexp_grad, compute_grad_X


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad_delta, compute_grad_X = _make_compute_addexp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_delta: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        g = grad_output[i]
        grad_delta[i] = compute_grad_delta(delta[i], g)
        grad_X[i] = compute_grad_X(delta[i], g)
    return implement


def _make_kernel_2d(dtype):
    compute_grad_delta, compute_grad_X = _make_compute_addexp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_delta: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        g = grad_output[i, j]
        grad_delta[i, j] = compute_grad_delta(delta[i, j], g)
        grad_X[i, j] = compute_grad_X(delta[i, j], g)
    return implement


def _make_kernel_3d(dtype):
    compute_grad_delta, compute_grad_X = _make_compute_addexp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_delta: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        g = grad_output[i, j, k]
        grad_delta[i, j, k] = compute_grad_delta(delta[i, j, k], g)
        grad_X[i, j, k] = compute_grad_X(delta[i, j, k], g)
    return implement


def _make_kernel_4d(dtype):
    compute_grad_delta, compute_grad_X = _make_compute_addexp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_delta: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        g = grad_output[i, j, k, l]
        grad_delta[i, j, k, l] = compute_grad_delta(delta[i, j, k, l], g)
        grad_X[i, j, k, l] = compute_grad_X(delta[i, j, k, l], g)
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

