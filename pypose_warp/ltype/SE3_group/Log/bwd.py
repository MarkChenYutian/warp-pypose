# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for SE3 Log.

Gradient computation:
    grad_X = [grad_output @ se3_Jl_inv(output), 0]

Where:
- output is the forward output [tau, phi] (6 elements)
- se3_Jl_inv is the 6x6 left Jacobian inverse for se3
- grad_X is [grad_t, grad_q] with grad_q having 0 in w component (7 elements total)
"""

import torch
import warp as wp
import typing as T

from ...common.warp_functions import se3_Jl_inv_wp_func
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
# Helper function for computing SE3_Log gradient
# =============================================================================

def _make_compute_se3_log_grad(dtype):
    """Create SE3 Log backward gradient function for the given dtype."""
    se3_Jl_inv_impl = se3_Jl_inv_wp_func(dtype)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_se3_log_grad(output: T.Any, grad_output: T.Any) -> T.Any:
        """
        Compute gradient of SE3_Log w.r.t. input.
        
        Args:
            output: Forward output [tau, phi] as 6D vector
            grad_output: Gradient w.r.t. output (6D vector)
            
        Returns:
            Gradient w.r.t. input as SE3 transform [grad_t, grad_q]
        """
        # Compute se3_Jl_inv (6x6 matrix)
        Jl_inv = se3_Jl_inv_impl(output)
        
        # grad = grad_output @ Jl_inv (row vector times matrix = row vector)
        # In warp, we compute Jl_inv^T @ grad_output (column form)
        grad_vec = wp.transpose(Jl_inv) @ grad_output
        
        # Extract grad_t (translation gradient) and grad_phi (rotation gradient)
        grad_t = vec3_ctor(grad_vec[0], grad_vec[1], grad_vec[2])
        
        # For quaternion gradient, the w component is 0
        grad_q = quat_ctor(grad_vec[3], grad_vec[4], grad_vec[5], dtype(0.0))
        
        return transform_ctor(grad_t, grad_q)
    
    return compute_se3_log_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad = _make_compute_se3_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        output: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad(output[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad = _make_compute_se3_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        output: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad(output[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad = _make_compute_se3_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        output: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad(output[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad = _make_compute_se3_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        output: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad(output[i, j, k, l], grad_output[i, j, k, l])
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

def SE3_Log_bwd(
    output: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of SE3_Log with respect to input.
    
    Args:
        output: Forward output (se3) of shape (..., 6)
        grad_output: Gradient w.r.t output of shape (..., 6)
        
    Returns:
        Gradient w.r.t input (SE3) of shape (..., 7)
    """
    # Prepare batch dimensions
    output, batch_info = prepare_batch_single(output)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = output.dtype
    device = output.device
    
    vec6_type = wp_vec6(dtype)
    transform_type = wp_transform(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    output = output.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    output_wp = wp.from_torch(output, dtype=vec6_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec6_type)
    
    grad_X_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=output_wp.device,
        inputs=[output_wp, grad_output_wp, grad_X_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info)

