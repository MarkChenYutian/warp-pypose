# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for SO3 AddExp: fused Exp(delta) * X operation.

This kernel combines:
1. Exponential map: so3 (axis-angle) -> SO3 (quaternion)
2. Quaternion multiplication

This is more efficient than separate Exp + Mul operations as it:
- Avoids intermediate memory allocation
- Reduces kernel launch overhead
- Fuses memory accesses
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.warp_functions import so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Helper function: fused Exp + Mul
# =============================================================================

def _make_add_exp_func(dtype):
    """Create fused Exp + Mul function for the given dtype."""
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.func
    def add_exp_func(delta: T.Any, X: T.Any) -> T.Any:
        """Compute Exp(delta) * X where delta is axis-angle and X is quaternion."""
        delta_quat = so3_exp_impl(delta)
        return wp.mul(delta_quat, X)
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

def SO3_AddExp_fwd(delta: torch.Tensor, X: pp.LieTensor) -> pp.LieTensor:
    """
    Compute fused Exp(delta) * X operation.
    
    This is equivalent to:
        Exp(delta) @ X
    where delta is in so3 (axis-angle, shape (..., 3)) and X is in SO3 (quaternion, shape (..., 4)).
    
    Args:
        delta: Tensor of shape (..., 3) - axis-angle representation (tangent space delta)
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        
    Returns:
        SO3 LieTensor of shape (..., 4) - result of Exp(delta) * X
    """
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
    vec3_type = wp_vec3_type(dtype)
    quat_type = wp_quat_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    delta_wp = wp.from_torch(delta_tensor, dtype=vec3_type)
    X_wp = wp.from_torch(X_tensor, dtype=quat_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=delta_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=quat_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=delta_wp.device,
        inputs=[delta_wp, X_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    from ... import warpSO3_type  # lazy import to avoid circular import
    return pp.LieTensor(out_tensor, ltype=warpSO3_type)

