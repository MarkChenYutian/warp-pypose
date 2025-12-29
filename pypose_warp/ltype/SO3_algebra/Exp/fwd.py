# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.warp_functions import so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = so3_exp_impl(x[i])
    return implement


def _make_kernel_2d(dtype):
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = so3_exp_impl(x[i, j])
    return implement


def _make_kernel_3d(dtype):
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = so3_exp_impl(x[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = so3_exp_impl(x[i, j, k, l])
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
}


# =============================================================================
# Main forward function
# =============================================================================

def so3_Exp_fwd(x: pp.LieTensor) -> pp.LieTensor:
    """
    Compute the exponential map of so3, mapping to SO3 Lie group (quaternion).
    
    Supports arbitrary batch dimensions (up to 4D).
    
    Args:
        x: so3 LieTensor of shape (..., 3) - axis-angle representation
        
    Returns:
        SO3 LieTensor of shape (..., 4) - quaternion representation
    """
    x_tensor = x.tensor()
    
    # Prepare batch dimensions
    x_tensor, batch_info = prepare_batch_single(x_tensor)
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec3_type = wp_vec3(dtype)
    quat_type = wp_quat(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    x_wp = wp.from_torch(x_tensor, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=x_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=quat_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    from ... import warpSO3_type  # lazy import to avoid circular import
    return pp.LieTensor(out_tensor, ltype=warpSO3_type)
