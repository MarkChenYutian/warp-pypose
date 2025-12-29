# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec3,
    wp_transform,
)


# =============================================================================
# SE3_Act Backward Pass
#
# Forward: out = t + R @ p
# Backward:
#   X_grad[:3] = grad_output (translation gradient)
#   X_grad[3:6] = cross(out, grad_output) (rotation Lie algebra gradient)
#   X_grad[6] = 0 (w component always zero)
#   p_grad = R^T @ grad_output
# =============================================================================


def _make_grad_funcs(dtype):
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_se3_grad_X(out: T.Any, g: T.Any) -> T.Any:
        """Compute SE3 gradient: (grad_output, cross(out, grad_output), 0)"""
        t_grad = g
        rot_grad = wp.cross(out, g)
        return transform_ctor(t_grad, quat_ctor(rot_grad[0], rot_grad[1], rot_grad[2], dtype(0.0)))

    @wp.func
    def compute_se3_grad_p(X: T.Any, g: T.Any) -> T.Any:
        """Compute point gradient: R^T @ grad_output"""
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        return wp.transpose(R) @ g

    return compute_se3_grad_X, compute_se3_grad_p


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_se3_grad_X(out[i], grad_output[i])
        grad_p[i] = compute_se3_grad_p(X[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_se3_grad_X(out[i, j], grad_output[i, j])
        grad_p[i, j] = compute_se3_grad_p(X[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_se3_grad_X(out[i, j, k], grad_output[i, j, k])
        grad_p[i, j, k] = compute_se3_grad_p(X[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_se3_grad_X(out[i, j, k, l], grad_output[i, j, k, l])
        grad_p[i, j, k, l] = compute_se3_grad_p(X[i, j, k, l], grad_output[i, j, k, l])
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

def SE3_Act_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_Act.
    
    Args:
        X: SE3 tensor of shape (..., 7)
        out: Output from forward pass, shape (..., 3)
        grad_output: Gradient w.r.t. output, shape (..., 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 7)
        grad_p: Gradient w.r.t. p, shape (..., 3)
    """
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    transform_type = wp_transform(dtype)
    vec3_type = wp_vec3(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    out_wp = wp.from_torch(out, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
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
