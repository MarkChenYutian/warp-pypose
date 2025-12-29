# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for SE3 Mat: gradient of 4x4 transformation matrix w.r.t SE3 pose.

SE3 representation: [tx, ty, tz, qx, qy, qz, qw] (7 elements)

4x4 Transformation matrix:
    [ R  | t ]     [ R00 R01 R02 | tx ]
    [----+---]  =  [ R10 R11 R12 | ty ]
    [ 0  | 1 ]     [ R20 R21 R22 | tz ]
                   [  0   0   0  | 1  ]

Gradient derivation:
- grad_t = [grad_M[0,3], grad_M[1,3], grad_M[2,3]] (translation is directly in the matrix)
- grad_q uses the same formula as SO3_Mat_bwd applied to the 3x3 rotation submatrix
"""

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_mat44_type, wp_transform_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
)


# =============================================================================
# Dtype-specific constructors
# =============================================================================

_DTYPE_TO_VEC3_CTOR = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}

_DTYPE_TO_QUAT_CTOR = {
    wp.float16: wp.quath,
    wp.float32: wp.quatf,
    wp.float64: wp.quatd,
}

_DTYPE_TO_TRANSFORM_CTOR = {
    wp.float16: wp.transformh,
    wp.float32: wp.transformf,
    wp.float64: wp.transformd,
}


# =============================================================================
# Helper function for computing SE3_Mat gradient
#
# Uses the same quaternion gradient formula as SO3_Mat_bwd for the rotation part.
# Translation gradient is directly read from the matrix gradient.
# =============================================================================

def _make_compute_se3_mat_grad(dtype):
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    quat_ctor = _DTYPE_TO_QUAT_CTOR[dtype]
    transform_ctor = _DTYPE_TO_TRANSFORM_CTOR[dtype]
    
    @wp.func
    def compute_se3_mat_grad(X: T.Any, G: T.Any) -> T.Any:
        """
        Compute SE3 gradient from 4x4 matrix gradient.
        
        Args:
            X: SE3 transform [t, q]
            G: Gradient w.r.t. 4x4 matrix
            
        Returns:
            Gradient w.r.t. SE3 pose as transform type [grad_t, grad_q]
        """
        # Extract quaternion from transform
        q = wp.transform_get_rotation(X)
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        
        # Translation gradient: directly from matrix gradient column 3 (indices 0,1,2)
        grad_tx = G[0, 3]
        grad_ty = G[1, 3]
        grad_tz = G[2, 3]
        
        # Quaternion gradient: same formula as SO3_Mat_bwd using 3x3 rotation submatrix
        # Symmetric and antisymmetric combinations
        G01_plus_G10 = G[0, 1] + G[1, 0]
        G02_plus_G20 = G[0, 2] + G[2, 0]
        G12_plus_G21 = G[1, 2] + G[2, 1]
        
        G21_minus_G12 = G[2, 1] - G[1, 2]
        G02_minus_G20 = G[0, 2] - G[2, 0]
        G10_minus_G01 = G[1, 0] - G[0, 1]
        
        grad_qx = dtype(4.0) * x * G[0, 0] + dtype(2.0) * y * G01_plus_G10 + dtype(2.0) * z * G02_plus_G20 + dtype(2.0) * w * G21_minus_G12
        grad_qy = dtype(2.0) * x * G01_plus_G10 + dtype(4.0) * y * G[1, 1] + dtype(2.0) * z * G12_plus_G21 + dtype(2.0) * w * G02_minus_G20
        grad_qz = dtype(2.0) * x * G02_plus_G20 + dtype(2.0) * y * G12_plus_G21 + dtype(4.0) * z * G[2, 2] + dtype(2.0) * w * G10_minus_G01
        grad_qw = dtype(4.0) * w * (G[0, 0] + G[1, 1] + G[2, 2]) + dtype(2.0) * x * G21_minus_G12 + dtype(2.0) * y * G02_minus_G20 + dtype(2.0) * z * G10_minus_G01
        
        # Construct gradient transform [grad_t, grad_q] using dtype-specific constructors
        grad_t = vec3_ctor(grad_tx, grad_ty, grad_tz)
        grad_q = quat_ctor(grad_qx, grad_qy, grad_qz, grad_qw)
        
        return transform_ctor(grad_t, grad_q)
    
    return compute_se3_mat_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad(X[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad(X[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad(X[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad(X[i, j, k, l], grad_output[i, j, k, l])
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
}


# =============================================================================
# Main backward function
# =============================================================================

def SE3_Mat_bwd(
    X: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for SE3_Mat.
    
    Args:
        X: SE3 tensor of shape (..., 7) - [tx, ty, tz, qx, qy, qz, qw]
        grad_output: Gradient w.r.t. output matrix, shape (..., 4, 4)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 7)
    """
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    transform_type = wp_transform_type(dtype)
    mat44_type = wp_mat44_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=mat44_type)
    
    # Allocate output gradient
    grad_X_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, grad_output_wp, grad_X_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info)

