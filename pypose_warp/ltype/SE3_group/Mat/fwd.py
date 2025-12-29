# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for SE3 Mat: converts SE3 pose to 4x4 transformation matrix.

SE3 representation in PyPose: [tx, ty, tz, qx, qy, qz, qw] (7 elements)

4x4 Transformation matrix:
    [ R  | t ]     [ R00 R01 R02 | tx ]
    [----+---]  =  [ R10 R11 R12 | ty ]
    [ 0  | 1 ]     [ R20 R21 R22 | tz ]
                   [  0   0   0  | 1  ]

This is more efficient than PyPose's default implementation which uses:
    I = eye(4); return X.unsqueeze(-2).Act(I).transpose(-1,-2)
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_mat44_type, wp_transform_type
from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Helper function: SE3 to 4x4 transformation matrix
# =============================================================================

def _make_se3_to_mat44(dtype):
    """Create SE3 to 4x4 matrix conversion function for the given dtype."""
    
    @wp.func
    def se3_to_mat44(X: T.Any) -> T.Any:
        """
        Convert SE3 transform to 4x4 transformation matrix.
        
        Args:
            X: warp transform (translation, quaternion)
            
        Returns:
            4x4 transformation matrix
        """
        # Extract translation and rotation
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        
        # Convert quaternion to 3x3 rotation matrix
        R = wp.quat_to_matrix(q)
        
        # Build 4x4 matrix element by element
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
    
    return se3_to_mat44


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    se3_to_mat44_impl = _make_se3_to_mat44(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_to_mat44_impl(X[i])
    return implement


def _make_kernel_2d(dtype):
    se3_to_mat44_impl = _make_se3_to_mat44(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_to_mat44_impl(X[i, j])
    return implement


def _make_kernel_3d(dtype):
    se3_to_mat44_impl = _make_se3_to_mat44(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_to_mat44_impl(X[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    se3_to_mat44_impl = _make_se3_to_mat44(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_to_mat44_impl(X[i, j, k, l])
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

def SE3_Mat_fwd(X: pp.LieTensor) -> torch.Tensor:
    """
    Convert SE3 pose to 4x4 transformation matrix.
    
    This is more efficient than PyPose's default implementation which uses:
        I = eye(4); return X.unsqueeze(-2).Act(I).transpose(-1,-2)
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [tx, ty, tz, qx, qy, qz, qw]
        
    Returns:
        Transformation matrix of shape (..., 4, 4)
    """
    from ...common.kernel_utils import TORCH_TO_WP_SCALAR
    
    X_tensor = X.tensor()
    
    # Prepare batch dimensions
    X_tensor, batch_info = prepare_batch_single(X_tensor)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    mat44_type = wp_mat44_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # PyPose SE3: [tx, ty, tz, qx, qy, qz, qw]
    # Warp transform: (translation, quaternion) where quat is [x, y, z, w]
    # The memory layout matches, so we can directly reinterpret
    X_wp = wp.from_torch(X_tensor.contiguous(), dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 4, 4), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat44_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)

