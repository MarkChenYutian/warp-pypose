# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.warp_functions import so3_Jl
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Backward kernel for so3_Exp
#
# The backward pass uses the left Jacobian Jl:
#   grad_input = grad_output[..., :-1] @ Jl(input)
# =============================================================================

def _make_compute_exp_grad(dtype):
    so3_Jl_impl = so3_Jl(dtype)
    
    @wp.func
    def compute_exp_grad(x: T.Any, grad_quat: T.Any) -> T.Any:
        """Compute grad_x for so3_Exp backward: grad_quat[:-1] @ Jl."""
        Jl = so3_Jl_impl(x)
        grad_xyz = wp.vector(grad_quat[0], grad_quat[1], grad_quat[2], dtype=dtype)
        return wp.transpose(Jl) @ grad_xyz
    return compute_exp_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_exp_grad_impl = _make_compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_x: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_x[i] = compute_exp_grad_impl(x[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_exp_grad_impl = _make_compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_x: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_x[i, j] = compute_exp_grad_impl(x[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_exp_grad_impl = _make_compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_x: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_x[i, j, k] = compute_exp_grad_impl(x[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_exp_grad_impl = _make_compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_x: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_x[i, j, k, l] = compute_exp_grad_impl(x[i, j, k, l], grad_output[i, j, k, l])
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

def so3_Exp_bwd(
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of so3_Exp with respect to input axis-angle vector.
    
    Args:
        x: Forward input (so3 axis-angle) of shape (..., 3)
        grad_output: Gradient w.r.t output quaternion of shape (..., 4)
        
    Returns:
        Gradient w.r.t input axis-angle of shape (..., 3)
    """
    # Prepare batch dimensions
    x, batch_info = prepare_batch_single(x)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = x.dtype
    device = x.device
    
    vec3_type = wp_vec3(dtype)
    quat_type = wp_quat(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    x = x.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    x_wp = wp.from_torch(x, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=quat_type)
    
    grad_x_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=device)
    grad_x_wp = wp.from_torch(grad_x_tensor, dtype=vec3_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, grad_output_wp, grad_x_wp],
    )
    
    return finalize_output(grad_x_tensor, batch_info)
