# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for SE3 AddExp: fused Exp(delta) * X operation.

This kernel combines:
1. Exponential map: se3 (twist) -> SE3 (pose)
2. SE3 multiplication

This is more efficient than separate Exp + Mul operations as it:
- Avoids intermediate memory allocation
- Reduces kernel launch overhead
- Fuses memory accesses

SE3 representation: [tx, ty, tz, qx, qy, qz, qw] (7 elements)
se3 representation: [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z] (6 elements)
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.warp_functions import so3_Jl, so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_TRANSFORM,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec6,
    wp_transform,
)


# =============================================================================
# Helper function: fused se3 Exp + SE3 Mul
# =============================================================================

def _make_add_exp_func(dtype):
    """Create fused Exp + Mul function for the given dtype."""
    so3_Jl_impl = so3_Jl(dtype)
    so3_exp_impl = so3_exp_wp_func(dtype)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def add_exp_func(delta: T.Any, X: T.Any) -> T.Any:
        """
        Compute Exp(delta) * X where delta is se3 twist and X is SE3 pose.
        
        Args:
            delta: se3 twist [tau, phi] as 6D vector
            X: SE3 pose [t, q] as transform
            
        Returns:
            SE3 pose [t_out, q_out] as transform
        """
        # Extract tau and phi from delta
        tau = vec3_ctor(delta[0], delta[1], delta[2])
        phi = vec3_ctor(delta[3], delta[4], delta[5])
        
        # Compute se3_Exp(delta):
        # t_exp = so3_Jl(phi) @ tau
        # q_exp = so3_Exp(phi)
        Jl = so3_Jl_impl(phi)
        t_exp = Jl @ tau
        q_exp = so3_exp_impl(phi)
        
        # SE3 multiplication: out = Exp(delta) * X
        # t_out = t_exp + R_exp @ t_X
        # q_out = q_exp * q_X
        t_X = wp.transform_get_translation(X)
        q_X = wp.transform_get_rotation(X)
        
        R_exp = wp.quat_to_matrix(q_exp)
        t_out = t_exp + R_exp @ t_X
        q_out = wp.mul(q_exp, q_X)
        
        return transform_ctor(t_out, q_out)
    
    return add_exp_func


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    add_exp_impl = _make_add_exp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=1),
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = add_exp_impl(delta[i], X[i])
    return implement


def _make_kernel_2d(dtype):
    add_exp_impl = _make_add_exp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=2),
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = add_exp_impl(delta[i, j], X[i, j])
    return implement


def _make_kernel_3d(dtype):
    add_exp_impl = _make_add_exp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=3),
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = add_exp_impl(delta[i, j, k], X[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    add_exp_impl = _make_add_exp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        delta: wp.array(dtype=T.Any, ndim=4),
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = add_exp_impl(delta[i, j, k, l], X[i, j, k, l])
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

def SE3_AddExp_fwd(delta: torch.Tensor, X: pp.LieTensor) -> pp.LieTensor:
    """
    Compute fused Exp(delta) * X operation for SE3.
    
    This is equivalent to:
        Exp(delta) @ X
    where delta is in se3 (twist, shape (..., 6)) and X is in SE3 (pose, shape (..., 7)).
    
    Args:
        delta: Tensor of shape (..., 6) - se3 twist representation [tau, phi]
        X: SE3 LieTensor of shape (..., 7) - pose representation [t, q]
        
    Returns:
        SE3 LieTensor of shape (..., 7) - result of Exp(delta) * X
    """
    from ... import warpSE3_type  # lazy import to avoid circular import
    
    # Handle LieTensor delta input
    delta_tensor = delta.tensor() if hasattr(delta, 'tensor') else delta
    X_tensor = X.tensor()
    
    # Ensure shapes match for batch dimensions
    assert delta_tensor.shape[:-1] == X_tensor.shape[:-1], \
        f"Batch shapes must match: delta {delta_tensor.shape[:-1]} vs X {X_tensor.shape[:-1]}"
    
    # Prepare batch dimensions
    delta_tensor, batch_info = prepare_batch_single(delta_tensor)
    
    if batch_info.squeeze_output:
        X_tensor = X_tensor.unsqueeze(0)
    
    # Get warp types based on dtype
    dtype = delta_tensor.dtype
    vec6_type = wp_vec6(dtype)
    transform_type = wp_transform(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    delta_wp = wp.from_torch(delta_tensor.contiguous(), dtype=vec6_type)
    X_wp = wp.from_torch(X_tensor.contiguous(), dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=delta_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=transform_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=delta_wp.device,
        inputs=[delta_wp, X_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    return pp.LieTensor(out_tensor, ltype=warpSE3_type)

