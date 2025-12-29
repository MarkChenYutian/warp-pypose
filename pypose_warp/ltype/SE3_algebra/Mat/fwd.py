# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for se3 Mat: converts se3 twist to 4x4 transformation matrix.

se3 representation: [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z] (6 elements)

This is equivalent to:
    se3.Exp().matrix()

4x4 Transformation matrix:
    [ R  | t ]     [ R00 R01 R02 | tx ]
    [----+---]  =  [ R10 R11 R12 | ty ]
    [ 0  | 1 ]     [ R20 R21 R22 | tz ]
                   [  0   0   0  | 1  ]

where:
- R = so3_Exp(phi).matrix() - 3x3 rotation matrix
- t = so3_Jl(phi) @ tau - translation vector
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_mat44_type
from ...common.warp_functions import so3_Jl, so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Dtype-specific vector constructors
# =============================================================================

_DTYPE_TO_VEC3_CTOR = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}


# =============================================================================
# Helper function: se3 (twist) -> 4x4 transformation matrix
# =============================================================================

def _make_se3_mat(dtype):
    """Create se3 to 4x4 matrix conversion function for the given dtype."""
    so3_Jl_impl = so3_Jl(dtype)
    so3_exp_impl = so3_exp_wp_func(dtype)
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    
    @wp.func
    def se3_mat(x: T.Any) -> T.Any:
        """
        Convert se3 twist to 4x4 transformation matrix.
        
        Args:
            x: se3 twist [tau, phi] as 6D vector
            
        Returns:
            4x4 transformation matrix
        """
        # Extract tau and phi
        tau = vec3_ctor(x[0], x[1], x[2])
        phi = vec3_ctor(x[3], x[4], x[5])
        
        # Compute t = so3_Jl(phi) @ tau
        Jl = so3_Jl_impl(phi)
        t = Jl @ tau
        
        # Compute quaternion and then rotation matrix
        q = so3_exp_impl(phi)
        R = wp.quat_to_matrix(q)
        
        # Build 4x4 transformation matrix
        M = wp.matrix(shape=(4, 4), dtype=dtype)
        
        # Rotation block (top-left 3x3)
        M[0, 0] = R[0, 0]
        M[0, 1] = R[0, 1]
        M[0, 2] = R[0, 2]
        M[1, 0] = R[1, 0]
        M[1, 1] = R[1, 1]
        M[1, 2] = R[1, 2]
        M[2, 0] = R[2, 0]
        M[2, 1] = R[2, 1]
        M[2, 2] = R[2, 2]
        
        # Translation column (top-right 3x1)
        M[0, 3] = t[0]
        M[1, 3] = t[1]
        M[2, 3] = t[2]
        
        # Bottom row [0, 0, 0, 1]
        M[3, 0] = dtype(0.0)
        M[3, 1] = dtype(0.0)
        M[3, 2] = dtype(0.0)
        M[3, 3] = dtype(1.0)
        
        return M
    
    return se3_mat


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_mat_impl = _make_se3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_mat_impl(x[i])
    return implement


def _make_kernel_2d(dtype):
    se3_mat_impl = _make_se3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_mat_impl(x[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_mat_impl = _make_se3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_mat_impl(x[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_mat_impl = _make_se3_mat(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_mat_impl(x[i, j, k, l])
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
}


# =============================================================================
# Warp type for 6D vector
# =============================================================================

def _wp_vec6_type(dtype: torch.dtype):
    match dtype:
        case torch.float64: return wp.types.vector(6, wp.float64)
        case torch.float32: return wp.types.vector(6, wp.float32)
        case torch.float16: return wp.types.vector(6, wp.float16)
        case _: raise NotImplementedError()


# =============================================================================
# Main forward function
# =============================================================================

def se3_Mat_fwd(x: pp.LieTensor) -> torch.Tensor:
    """
    Convert se3 twist to 4x4 transformation matrix.
    
    This is equivalent to PyPose's se3Type.matrix() method:
        se3.Exp().matrix()
    
    But more efficient as it computes directly in a single fused kernel.
    
    Args:
        x: se3 LieTensor of shape (..., 6) - [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z]
        
    Returns:
        Transformation matrix of shape (..., 4, 4)
    """
    x_tensor = x.tensor() if hasattr(x, 'tensor') else x
    
    # Prepare batch dimensions
    x_tensor, batch_info = prepare_batch_single(x_tensor)
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec6_type = _wp_vec6_type(dtype)
    mat44_type = wp_mat44_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    x_wp = wp.from_torch(x_tensor.contiguous(), dtype=vec6_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 4, 4), dtype=dtype, device=x_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat44_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)

