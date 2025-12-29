# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Backward kernels for SO3_Act
# 
# Given: grad_output (gradient w.r.t. output of forward pass)
# Compute: 
#   - X_grad: gradient w.r.t. quaternion X (shape ..., 4)
#   - p_grad: gradient w.r.t. point p (shape ..., 3)
#
# From PyPose:
#   X_grad[:3] = grad_output @ skew(-out)  (Jacobian w.r.t. Lie algebra)
#   X_grad[3] = 0  (w component has zero gradient)
#   p_grad = grad_output @ R  (where R = rotation matrix from X)
#
# OPTIMIZATION: Uses wp.quat_rotate_inv(q, g) instead of transpose(quat_to_matrix(q)) @ g.
# This is mathematically equivalent but more efficient.
# =============================================================================

def _make_compute_grad_X(dtype):
    @wp.func
    def compute_grad_X(o: T.Any, g: T.Any) -> T.Any:
        """Compute quaternion gradient: (cross(out, grad), 0)"""
        c = wp.cross(o, g)
        return wp.quaternion(c[0], c[1], c[2], dtype(0.0))
    return compute_grad_X


def _make_kernel_1d(dtype):
    compute_grad_X = _make_compute_grad_X(dtype)

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
        grad_p[i] = wp.quat_rotate_inv(q, g)
        grad_X[i] = compute_grad_X(o, g)
    return implement


def _make_kernel_2d(dtype):
    compute_grad_X = _make_compute_grad_X(dtype)

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
        grad_p[i, j] = wp.quat_rotate_inv(q, g)
        grad_X[i, j] = compute_grad_X(o, g)
    return implement


def _make_kernel_3d(dtype):
    compute_grad_X = _make_compute_grad_X(dtype)

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
        grad_p[i, j, k] = wp.quat_rotate_inv(q, g)
        grad_X[i, j, k] = compute_grad_X(o, g)
    return implement


def _make_kernel_4d(dtype):
    compute_grad_X = _make_compute_grad_X(dtype)

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
        grad_p[i, j, k, l] = wp.quat_rotate_inv(q, g)
        grad_X[i, j, k, l] = compute_grad_X(o, g)
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

def SO3_Act_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SO3_Act.
    
    Args:
        X: Quaternion tensor of shape (..., 4) - expanded to broadcast shape
        out: Output from forward pass, shape (..., 3) (saved from forward)
        grad_output: Gradient w.r.t. output, shape (..., 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 4)
        grad_p: Gradient w.r.t. p, shape (..., 3)
    """
    # Prepare batch dimensions
    X_tensor, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X_tensor.dtype
    device = X_tensor.device
    quat_type = wp_quat(dtype)
    vec3_type = wp_vec3(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    X_tensor = X_tensor.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_tensor, dtype=quat_type)
    out_wp = wp.from_torch(out, dtype=vec3_type)
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
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info), finalize_output(grad_p_tensor, batch_info)
