# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_quat,
    wp_mat33,
)


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# Each kernel uses wp.quat_to_matrix() which efficiently converts quaternion
# to 3x3 rotation matrix.
# =============================================================================

def _make_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = wp.quat_to_matrix(X[i])
    return implement


def _make_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = wp.quat_to_matrix(X[i, j])
    return implement


def _make_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = wp.quat_to_matrix(X[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = wp.quat_to_matrix(X[i, j, k, l])
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

def SO3_Mat_fwd(X: pp.LieTensor) -> torch.Tensor:
    """
    Convert SO3 quaternion to 3x3 rotation matrix.
    
    This is more efficient than PyPose's default implementation which uses:
        I = eye(3); return X.Act(I).transpose(-1,-2)
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation (x, y, z, w)
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions
    X_tensor, batch_info = prepare_batch_single(X_tensor)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat(dtype)
    mat33_type = wp_mat33(dtype)
    
    # Convert to warp array
    X_wp = wp.from_torch(X_tensor.contiguous(), dtype=quat_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat33_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, None)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
