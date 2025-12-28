# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.warp_functions import SO3_log_wp_func
from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = SO3_log_wp_func(X[i])
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = SO3_log_wp_func(X[i, j])
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = SO3_log_wp_func(X[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = SO3_log_wp_func(X[i, j, k, l])
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

def SO3_Log_fwd(X: pp.LieTensor) -> pp.LieTensor:
    """
    Compute the logarithm map of SO3, mapping to so3 Lie algebra.
    
    Supports arbitrary batch dimensions (up to 4D).
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        
    Returns:
        so3 LieTensor of shape (..., 3) - axis-angle representation
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions
    X_tensor, batch_info = prepare_batch_single(X_tensor)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    
    # Convert to warp array
    X_wp = wp.from_torch(X_tensor, dtype=quat_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Get kernel and launch (dtype is ignored for SO3_log since it uses wp.quat_to_axis_angle)
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, None)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    out_tensor = finalize_output(out_tensor, batch_info)
    
    from ... import warpso3_type  # lazy import to avoid circular import
    return pp.LieTensor(out_tensor, ltype=warpso3_type)
