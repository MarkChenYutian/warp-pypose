# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_mat33_type


# =============================================================================
# Kernels for different batch dimensions (1D to 4D)
# Warp requires fixed ndim at compile time, so we need separate kernels.
# Each kernel uses wp.quat_to_matrix() which efficiently converts quaternion
# to 3x3 rotation matrix.
# =============================================================================

@wp.kernel(enable_backward=False)
def SO3_Mat_fwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    out: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    out[i] = wp.quat_to_matrix(X[i])


@wp.kernel(enable_backward=False)
def SO3_Mat_fwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    out: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid()  # type: ignore
    out[i, j] = wp.quat_to_matrix(X[i, j])


@wp.kernel(enable_backward=False)
def SO3_Mat_fwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    out: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid()  # type: ignore
    out[i, j, k] = wp.quat_to_matrix(X[i, j, k])


@wp.kernel(enable_backward=False)
def SO3_Mat_fwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    out: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid()  # type: ignore
    out[i, j, k, l] = wp.quat_to_matrix(X[i, j, k, l])


# =============================================================================
# Kernel selection map: (ndim) -> kernel function
# =============================================================================

_SO3_Mat_kernels = {
    1: SO3_Mat_fwd_kernel_1d,
    2: SO3_Mat_fwd_kernel_2d,
    3: SO3_Mat_fwd_kernel_3d,
    4: SO3_Mat_fwd_kernel_4d,
}


# =============================================================================
# Main forward function with arbitrary dimension support
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
    
    # Get batch shape (everything except last dim)
    batch_shape = X_tensor.shape[:-1]
    
    ndim = len(batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    mat33_type = wp_mat33_type(dtype)
    
    # Convert to warp array
    X_wp = wp.from_torch(X_tensor.contiguous(), dtype=quat_type)
    
    # Create output tensor and warp array
    # Output shape is (..., 3, 3) but warp mat33 is stored as a single element
    out_tensor = torch.empty((*batch_shape, 3, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat33_type)
    
    # Launch kernel with multi-dimensional grid
    wp.launch(
        kernel=_SO3_Mat_kernels[ndim],
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# This ensures warp compiles specialized versions for each type combination.
# =============================================================================

__SO3_Mat_fwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_Mat_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        out=wp.array(dtype=wp.mat33h, ndim=1),
    )),
    wp.overload(SO3_Mat_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        out=wp.array(dtype=wp.mat33f, ndim=1),
    )),
    wp.overload(SO3_Mat_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        out=wp.array(dtype=wp.mat33d, ndim=1),
    )),
    
    # 2D kernels
    wp.overload(SO3_Mat_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        out=wp.array(dtype=wp.mat33h, ndim=2),
    )),
    wp.overload(SO3_Mat_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        out=wp.array(dtype=wp.mat33f, ndim=2),
    )),
    wp.overload(SO3_Mat_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        out=wp.array(dtype=wp.mat33d, ndim=2),
    )),
    
    # 3D kernels
    wp.overload(SO3_Mat_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        out=wp.array(dtype=wp.mat33h, ndim=3),
    )),
    wp.overload(SO3_Mat_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        out=wp.array(dtype=wp.mat33f, ndim=3),
    )),
    wp.overload(SO3_Mat_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        out=wp.array(dtype=wp.mat33d, ndim=3),
    )),
    
    # 4D kernels
    wp.overload(SO3_Mat_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        out=wp.array(dtype=wp.mat33h, ndim=4),
    )),
    wp.overload(SO3_Mat_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        out=wp.array(dtype=wp.mat33f, ndim=4),
    )),
    wp.overload(SO3_Mat_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        out=wp.array(dtype=wp.mat33d, ndim=4),
    )),
]

