# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for SE3 AddExp: fused Exp(delta) * X operation.

Given Y = Exp(delta) * X, we need gradients w.r.t. delta and X.

The chain rule gives us:
- grad_delta = se3_Jl(delta)^T @ grad_Y[:-1]
- grad_X = Adj(Exp(delta))^T @ grad_Y[:-1]

For SE3 Mul Y = A * X:
- grad_X[:3] = R_A^T @ grad_t
- grad_X[3:6] = R_A^T @ (cross(grad_t, t_A) + grad_r)
- grad_X[6] = 0

where A = Exp(delta), and t_A, R_A are the translation and rotation from A.
"""

import torch
import warp as wp
import typing as T

from ...common.warp_functions import se3_Jl_wp_func, so3_Jl, so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec6,
    wp_transform,
)


# =============================================================================
# Helper functions for computing AddExp backward
# =============================================================================

def _make_compute_addexp_grad(dtype):
    """Create gradient computation functions for the given dtype."""
    se3_Jl_impl = se3_Jl_wp_func(dtype)
    so3_Jl_impl = so3_Jl(dtype)
    so3_exp_impl = so3_exp_wp_func(dtype)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_grad_delta(delta: T.Any, grad: T.Any) -> T.Any:
        """
        Compute grad_delta for AddExp backward.
        
        grad_delta = se3_Jl(delta)^T @ grad[:-1]
        """
        Jl = se3_Jl_impl(delta)
        
        # Extract grad components (first 6 elements of SE3 gradient)
        grad_t = wp.transform_get_translation(grad)
        grad_r = vec3_ctor(grad[3], grad[4], grad[5])
        
        # Build 6D gradient vector
        grad_6d = wp.vector(
            grad_t[0], grad_t[1], grad_t[2],
            grad_r[0], grad_r[1], grad_r[2],
            dtype=dtype
        )
        
        # grad_delta = Jl^T @ grad_6d
        return wp.transpose(Jl) @ grad_6d
    
    @wp.func
    def compute_grad_X(delta: T.Any, grad: T.Any) -> T.Any:
        """
        Compute grad_X for AddExp backward.
        
        This is the gradient w.r.t. X in Y = Exp(delta) * X.
        Using SE3 Mul backward formula for the second operand:
        
        grad_X = Adj(Exp(delta))^T @ grad[:-1]
        
        Where:
        - grad_X[:3] = R^T @ grad_t
        - grad_X[3:6] = R^T @ (cross(grad_t, t) + grad_r)
        - grad_X[6] = 0
        
        Here R, t are from Exp(delta).
        """
        # Extract tau and phi from delta
        tau = vec3_ctor(delta[0], delta[1], delta[2])
        phi = vec3_ctor(delta[3], delta[4], delta[5])
        
        # Compute Exp(delta) to get t_exp and R_exp
        Jl = so3_Jl_impl(phi)
        t_exp = Jl @ tau
        q_exp = so3_exp_impl(phi)
        R_exp = wp.quat_to_matrix(q_exp)
        RT = wp.transpose(R_exp)
        
        # Extract gradient components
        grad_t = wp.transform_get_translation(grad)
        grad_r = vec3_ctor(grad[3], grad[4], grad[5])
        
        # Compute grad_X[:3] = R^T @ grad_t
        grad_X_t = RT @ grad_t
        
        # Compute grad_X[3:6] = R^T @ (cross(grad_t, t_exp) + grad_r)
        grad_X_r = RT @ (wp.cross(grad_t, t_exp) + grad_r)
        
        # grad_X[6] = 0
        return transform_ctor(
            grad_X_t,
            quat_ctor(grad_X_r[0], grad_X_r[1], grad_X_r[2], dtype(0.0))
        )
    
    return compute_grad_delta, compute_grad_X


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

def SE3_AddExp_bwd(
    delta: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SE3_AddExp with respect to inputs.
    
    Given Y = Exp(delta) * X, computes:
    - grad_delta: Gradient w.r.t. delta (se3 twist)
    - grad_X: Gradient w.r.t. X (SE3 pose)
    
    Args:
        delta: Forward input (se3 twist) of shape (..., 6)
        grad_output: Gradient w.r.t. output SE3 pose of shape (..., 7)
        
    Returns:
        Tuple of (grad_delta, grad_X):
        - grad_delta: shape (..., 6)
        - grad_X: shape (..., 7)
    """
    # Prepare batch dimensions
    delta, batch_info = prepare_batch_single(delta)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = delta.dtype
    device = delta.device
    
    vec6_type = wp_vec6(dtype)
    transform_type = wp_transform(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    delta = delta.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    delta_wp = wp.from_torch(delta, dtype=vec6_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=transform_type)
    
    # Create output tensors
    grad_delta_tensor = torch.empty((*batch_info.shape, 6), dtype=dtype, device=device)
    grad_X_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=device)
    
    grad_delta_wp = wp.from_torch(grad_delta_tensor, dtype=vec6_type)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=delta_wp.device,
        inputs=[delta_wp, grad_output_wp, grad_delta_wp, grad_X_wp],
    )
    
    return finalize_output(grad_delta_tensor, batch_info), finalize_output(grad_X_tensor, batch_info)

