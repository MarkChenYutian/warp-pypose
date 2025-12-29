# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for se3 Exp: se3 Lie algebra -> SE3 Lie group.

se3 representation: [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z] (6 elements)
SE3 representation: [tx, ty, tz, qx, qy, qz, qw] (7 elements)

Algorithm:
1. Extract tau = [tau_x, tau_y, tau_z] (translation in Lie algebra)
2. Extract phi = [phi_x, phi_y, phi_z] (rotation axis-angle)
3. Compute Jl = so3_Jl(phi) - 3x3 left Jacobian
4. Compute t = Jl @ tau - translation in SE3
5. Compute q = so3_Exp(phi) - quaternion from axis-angle
6. Return [t, q] (7 elements)
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type
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

_DTYPE_TO_TRANSFORM_CTOR = {
    wp.float16: wp.transformh,
    wp.float32: wp.transformf,
    wp.float64: wp.transformd,
}


# =============================================================================
# Helper function: se3 Exp forward
# =============================================================================

def _make_se3_exp(dtype):
    """Create se3 Exp forward function for the given dtype."""
    so3_Jl_impl = so3_Jl(dtype)
    so3_exp_impl = so3_exp_wp_func(dtype)
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    transform_ctor = _DTYPE_TO_TRANSFORM_CTOR[dtype]
    
    @wp.func
    def se3_exp(x: T.Any) -> T.Any:
        """
        Compute se3 Exp: se3 Lie algebra -> SE3 Lie group.
        
        Args:
            x: se3 element [tau, phi] as 6D vector
            
        Returns:
            SE3 transform [t, q]
        """
        # Extract tau and phi
        tau = vec3_ctor(x[0], x[1], x[2])
        phi = vec3_ctor(x[3], x[4], x[5])
        
        # Compute Jl = so3_Jl(phi)
        Jl = so3_Jl_impl(phi)
        
        # Compute t = Jl @ tau
        t = Jl @ tau
        
        # Compute q = so3_Exp(phi)
        q = so3_exp_impl(phi)
        
        return transform_ctor(t, q)
    
    return se3_exp


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_exp_impl = _make_se3_exp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_exp_impl(x[i])
    return implement


def _make_kernel_2d(dtype):
    se3_exp_impl = _make_se3_exp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_exp_impl(x[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_exp_impl = _make_se3_exp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_exp_impl(x[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_exp_impl = _make_se3_exp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_exp_impl(x[i, j, k, l])
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

def se3_Exp_fwd(x: pp.LieTensor) -> pp.LieTensor:
    """
    Compute the exponential map of se3, mapping to SE3 Lie group.
    
    Args:
        x: se3 LieTensor of shape (..., 6) - [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z]
        
    Returns:
        SE3 LieTensor of shape (..., 7) - [tx, ty, tz, qx, qy, qz, qw]
    """
    from ...SE3_group import warpSE3_type
    
    x_tensor = x.tensor()
    
    # Prepare batch dimensions
    x_tensor, batch_info = prepare_batch_single(x_tensor)
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec6_type = _wp_vec6_type(dtype)
    transform_type = wp_transform_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    x_wp = wp.from_torch(x_tensor.contiguous(), dtype=vec6_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=x_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=transform_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    return pp.LieTensor(out_tensor, ltype=warpSE3_type)

