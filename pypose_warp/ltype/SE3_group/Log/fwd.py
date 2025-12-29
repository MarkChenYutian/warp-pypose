# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for SE3 Log: SE3 group -> se3 Lie algebra.

SE3 representation: [tx, ty, tz, qx, qy, qz, qw] (7 elements)
se3 representation: [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z] (6 elements)

Algorithm:
1. Extract quaternion q = [qx, qy, qz, qw] from SE3
2. Compute phi = SO3_Log(q) - rotation axis-angle (3 elements)
3. Compute Jl_inv = so3_Jl_inv(phi) - 3x3 left Jacobian inverse
4. Extract translation t = [tx, ty, tz] from SE3
5. Compute tau = Jl_inv @ t - translation in Lie algebra
6. Return [tau, phi] (6 elements)
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.warp_functions import SO3_log_wp_func, so3_Jl_inv
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec6,
    wp_transform,
)


# =============================================================================
# Helper function: SE3 Log forward
# =============================================================================

def _make_se3_log(dtype):
    """Create SE3 Log forward function for the given dtype."""
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    
    @wp.func
    def se3_log(X: T.Any) -> T.Any:
        """
        Compute SE3 Log: SE3 group -> se3 Lie algebra.
        
        Args:
            X: SE3 transform [t, q]
            
        Returns:
            se3 Lie algebra element [tau, phi] as 6D vector
        """
        # Extract translation and rotation
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        
        # Compute phi = SO3_Log(q)
        phi = SO3_log_wp_func(q)
        
        # Compute Jl_inv = so3_Jl_inv(phi)
        Jl_inv = so3_Jl_inv_impl(phi)
        
        # Compute tau = Jl_inv @ t
        tau = Jl_inv @ t
        
        # Return [tau, phi] as 6D vector
        return wp.vector(tau[0], tau[1], tau[2], phi[0], phi[1], phi[2], dtype=dtype)
    
    return se3_log


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_log_impl = _make_se3_log(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_log_impl(X[i])
    return implement


def _make_kernel_2d(dtype):
    se3_log_impl = _make_se3_log(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_log_impl(X[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_log_impl = _make_se3_log(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_log_impl(X[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_log_impl = _make_se3_log(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_log_impl(X[i, j, k, l])
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

def SE3_Log_fwd(X: pp.LieTensor) -> pp.LieTensor:
    """
    Compute the logarithm map of SE3, mapping to se3 Lie algebra.
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [tx, ty, tz, qx, qy, qz, qw]
        
    Returns:
        se3 LieTensor of shape (..., 6) - [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z]
    """
    from ...SE3_algebra import warpse3_type
    
    X_tensor = X.tensor()
    
    # Prepare batch dimensions
    X_tensor, batch_info = prepare_batch_single(X_tensor)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform(dtype)
    vec6_type = wp_vec6(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    X_wp = wp.from_torch(X_tensor.contiguous(), dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 6), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec6_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    # Return as se3 LieTensor
    import pypose as pp
    return pp.LieTensor(out_tensor, ltype=warpse3_type)

