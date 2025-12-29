# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for se3 Exp.

Gradient computation:
    grad_input = grad_output[..., :-1] @ se3_Jl(input)

Where:
- input is the forward input [tau, phi] (6 elements)
- se3_Jl is the 6x6 left Jacobian for se3
- grad_output[..., :-1] extracts the first 6 elements (ignoring qw gradient)
"""

import torch
import warp as wp
import typing as T

from ...common.warp_functions import se3_Jl_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec6,
    wp_transform,
)


# =============================================================================
# Helper function for computing se3_Exp gradient
# =============================================================================

def _make_compute_se3_exp_grad(dtype):
    """Create se3 Exp backward gradient function for the given dtype."""
    se3_Jl_impl = se3_Jl_wp_func(dtype)
    
    @wp.func
    def compute_se3_exp_grad(input: T.Any, grad_output: T.Any) -> T.Any:
        """
        Compute gradient of se3_Exp w.r.t. input.
        
        Args:
            input: Forward input [tau, phi] as 6D vector
            grad_output: Gradient w.r.t. output as SE3 transform [t, q]
            
        Returns:
            Gradient w.r.t. input as 6D vector
        """
        # Compute se3_Jl (6x6 matrix)
        Jl = se3_Jl_impl(input)
        
        # Extract gradient from transform (first 6 elements, skip qw)
        # grad_output is [tx, ty, tz, qx, qy, qz, qw] as transform
        t = wp.transform_get_translation(grad_output)
        q = wp.transform_get_rotation(grad_output)
        
        # Create 6D gradient vector [grad_t, grad_q_xyz]
        grad_6d = wp.vector(t[0], t[1], t[2], q[0], q[1], q[2], dtype=dtype)
        
        # grad_input = Jl^T @ grad_6d
        grad_input = wp.transpose(Jl) @ grad_6d
        
        return grad_input
    
    return compute_se3_exp_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad = _make_compute_se3_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        input: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_input: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_input[i] = compute_grad(input[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad = _make_compute_se3_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        input: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_input: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_input[i, j] = compute_grad(input[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad = _make_compute_se3_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        input: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_input: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_input[i, j, k] = compute_grad(input[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad = _make_compute_se3_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        input: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_input: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_input[i, j, k, l] = compute_grad(input[i, j, k, l], grad_output[i, j, k, l])
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

def se3_Exp_bwd(
    input: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of se3_Exp with respect to input.
    
    Args:
        input: Forward input (se3) of shape (..., 6)
        grad_output: Gradient w.r.t output (SE3) of shape (..., 7)
        
    Returns:
        Gradient w.r.t input (se3) of shape (..., 6)
    """
    # Prepare batch dimensions
    input, batch_info = prepare_batch_single(input)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = input.dtype
    device = input.device
    
    vec6_type = wp_vec6(dtype)
    transform_type = wp_transform(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    input = input.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    input_wp = wp.from_torch(input, dtype=vec6_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=transform_type)
    
    grad_input_tensor = torch.empty((*batch_info.shape, 6), dtype=dtype, device=device)
    grad_input_wp = wp.from_torch(grad_input_tensor, dtype=vec6_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=input_wp.device,
        inputs=[input_wp, grad_output_wp, grad_input_wp],
    )
    
    return finalize_output(grad_input_tensor, batch_info)

