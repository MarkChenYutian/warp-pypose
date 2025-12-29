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
    wp_mat33,
)


# =============================================================================
# Helper function: so3 (axis-angle) -> mat33 (rotation matrix)
# =============================================================================

def _make_so3_mat(dtype):
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.func
    def so3_mat(x: T.Any) -> T.Any:
        """Convert so3 (axis-angle) to rotation matrix via quaternion."""
        q = so3_exp_impl(x)
        return wp.quat_to_matrix(q)
    return so3_mat


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    so3_mat_impl = _make_so3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = so3_mat_impl(x[i])
    return implement


def _make_kernel_2d(dtype):
    so3_mat_impl = _make_so3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = so3_mat_impl(x[i, j])
    return implement


def _make_kernel_3d(dtype):
    so3_mat_impl = _make_so3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = so3_mat_impl(x[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    so3_mat_impl = _make_so3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = so3_mat_impl(x[i, j, k, l])
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

def so3_Mat_fwd(x: pp.LieTensor) -> torch.Tensor:
    """
    Convert so3 (axis-angle) to 3x3 rotation matrix.
    
    This is equivalent to PyPose's so3Type.matrix() method:
        X = input.Exp()
        I = eye(3)
        return X.Act(I).transpose(-1,-2)
    
    But more efficient as it computes directly in a single kernel.
    
    Args:
        x: so3 LieTensor of shape (..., 3) - axis-angle representation
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    x_tensor = x.tensor() if hasattr(x, 'tensor') else x
    
    # Prepare batch dimensions
    x_tensor, batch_info = prepare_batch_single(x_tensor)
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec3_type = wp_vec3(dtype)
    mat33_type = wp_mat33(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    x_wp = wp.from_torch(x_tensor.contiguous(), dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3, 3), dtype=dtype, device=x_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat33_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
