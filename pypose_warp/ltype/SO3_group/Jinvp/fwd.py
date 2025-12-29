# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.warp_functions import SO3_log_wp_func, so3_Jl_inv
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
    wp_vec3,
    wp_quat,
)


# =============================================================================
# Higher-order function to generate dtype-specific Jinvp computation
# =============================================================================

def _make_compute_jinvp(dtype):
    """Generate a dtype-specific Jinvp computation function."""
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    
    @wp.func
    def compute_jinvp(q: T.Any, p: T.Any) -> T.Any:
        """Compute Jl_inv(Log(X)) @ p."""
        so3 = SO3_log_wp_func(q)
        Jl_inv = so3_Jl_inv_impl(so3)
        return Jl_inv @ p
    return compute_jinvp


# =============================================================================
# Kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_jinvp_impl = _make_compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = compute_jinvp_impl(X[i], p[i])
    return implement


def _make_kernel_2d(dtype):
    compute_jinvp_impl = _make_compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = compute_jinvp_impl(X[i, j], p[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_jinvp_impl = _make_compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = compute_jinvp_impl(X[i, j, k], p[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_jinvp_impl = _make_compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = compute_jinvp_impl(X[i, j, k, l], p[i, j, k, l])
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

def SO3_Jinvp_fwd(X: pp.LieTensor, p: torch.Tensor) -> pp.LieTensor:
    """
    Compute Jinvp: Jl_inv(Log(X)) @ p
    
    Maps a tangent vector p through the inverse left Jacobian of the Log map.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        p: Tensor of shape (..., 3) - so3 tangent vector
        
    Returns:
        so3 LieTensor of shape (broadcast(...), 3) - transformed tangent vector
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions with broadcasting
    X_tensor, p, batch_info = prepare_batch_broadcast(X_tensor, p)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 4)
    p_expanded = p.expand(*batch_info.shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat(dtype)
    vec3_type = wp_vec3(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    from ... import warpso3_type  # lazy import to avoid circular import
    return pp.LieTensor(out_tensor, ltype=warpso3_type)
