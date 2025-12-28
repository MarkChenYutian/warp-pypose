# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.warp_functions import so3_Jl_inv
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


def _make_compute_log_grad(dtype):
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    
    @wp.func
    def compute_log_grad(x: T.Any, g: T.Any) -> T.Any:
        """Compute grad_X for SO3_Log backward: [g @ Jl_inv, 0]."""
        Jl_inv = so3_Jl_inv_impl(x)
        grad_vec = wp.transpose(Jl_inv) @ g
        return wp.quaternion(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    return compute_log_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_log_grad_impl = _make_compute_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_log_grad_impl(out[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_log_grad_impl = _make_compute_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_log_grad_impl(out[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_log_grad_impl = _make_compute_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_log_grad_impl(out[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_log_grad_impl = _make_compute_log_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_log_grad_impl(out[i, j, k, l], grad_output[i, j, k, l])
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

def SO3_Log_bwd(
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of SO3_Log with respect to input quaternion.
    
    Args:
        out: Forward output (so3 Lie algebra) of shape (..., 3)
        grad_output: Gradient w.r.t output of shape (..., 3)
        
    Returns:
        Gradient w.r.t input quaternion of shape (..., 4)
    """
    # Prepare batch dimensions
    out_tensor, batch_info = prepare_batch_single(out)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = out_tensor.dtype
    device = out_tensor.device
    
    vec3_type = wp_vec3_type(dtype)
    quat_type = wp_quat_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    out_tensor = out_tensor.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    grad_X_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=out_wp.device,
        inputs=[out_wp, grad_output_wp, grad_X_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info)
