# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# SE3_Inv Forward Pass
#
# Inverse of SE3 transform:
#   q_inv = conjugate(q)
#   t_inv = -R_inv @ t
# =============================================================================


def _make_se3_inv(dtype):
    @wp.func
    def se3_inv(X: T.Any) -> T.Any:
        """Compute the inverse of an SE3 transform."""
        return wp.transform_inverse(X)
    return se3_inv


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_inv(X[i])
    return implement


def _make_kernel_2d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_inv(X[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_inv(X[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_inv(X[i, j, k, l])
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

def SE3_Inv_fwd(X: pp.LieTensor) -> torch.Tensor:
    """
    Compute the inverse of SE3 transformation X.
    
    Args:
        X: SE3 LieTensor of shape (..., 7)
        
    Returns:
        Inverted SE3 tensor of shape (..., 7)
    """
    X_tensor = X.tensor()
    
    # Prepare batch dimensions
    X_tensor, batch_info = prepare_batch_single(X_tensor)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Ensure contiguous for warp
    X_tensor = X_tensor.contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_tensor, dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=transform_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
