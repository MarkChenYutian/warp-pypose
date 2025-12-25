# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_mat33_type

# =============================================================================
# Kernels for different batch dimensions (1D to 4D)
# =============================================================================

@wp.kernel(enable_backward=False)
def SO3_Adj_fwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    out: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    out[i] = wp.quat_to_matrix(X[i])


@wp.kernel(enable_backward=False)
def SO3_Adj_fwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    out: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid()  # type: ignore
    out[i, j] = wp.quat_to_matrix(X[i, j])


@wp.kernel(enable_backward=False)
def SO3_Adj_fwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    out: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid()  # type: ignore
    out[i, j, k] = wp.quat_to_matrix(X[i, j, k])


@wp.kernel(enable_backward=False)
def SO3_Adj_fwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    out: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid()  # type: ignore
    out[i, j, k, l] = wp.quat_to_matrix(X[i, j, k, l])


# =============================================================================
# Kernel selection map
# =============================================================================

_SO3_Adj_kernels = {
    1: SO3_Adj_fwd_kernel_1d,
    2: SO3_Adj_fwd_kernel_2d,
    3: SO3_Adj_fwd_kernel_3d,
    4: SO3_Adj_fwd_kernel_4d,
}


# =============================================================================
# Main forward function
# =============================================================================

def SO3_Adj_fwd(X: pp.LieTensor) -> torch.Tensor:
    """
    Compute the Adjoint representation of SO3 (rotation matrix).
    
    Supports arbitrary batch dimensions (up to 4D).
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        
    Returns:
        Tensor of shape (..., 3, 3) - rotation matrix
    """
    X_tensor = X.tensor()
    
    batch_shape = X_tensor.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        X_tensor = X_tensor.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    mat33_type = wp_mat33_type(dtype)
    
    X_wp = wp.from_torch(X_tensor, dtype=quat_type)
    
    out_tensor = torch.empty((*batch_shape, 3, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat33_type)
    
    wp.launch(
        kernel=_SO3_Adj_kernels[ndim],
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# =============================================================================

__SO3_Adj_fwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_Adj_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        out=wp.array(dtype=wp.mat33h, ndim=1)
    )),
    wp.overload(SO3_Adj_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        out=wp.array(dtype=wp.mat33f, ndim=1)
    )),
    wp.overload(SO3_Adj_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        out=wp.array(dtype=wp.mat33d, ndim=1)
    )),
    
    # 2D kernels
    wp.overload(SO3_Adj_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        out=wp.array(dtype=wp.mat33h, ndim=2)
    )),
    wp.overload(SO3_Adj_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        out=wp.array(dtype=wp.mat33f, ndim=2)
    )),
    wp.overload(SO3_Adj_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        out=wp.array(dtype=wp.mat33d, ndim=2)
    )),
    
    # 3D kernels
    wp.overload(SO3_Adj_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        out=wp.array(dtype=wp.mat33h, ndim=3)
    )),
    wp.overload(SO3_Adj_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        out=wp.array(dtype=wp.mat33f, ndim=3)
    )),
    wp.overload(SO3_Adj_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        out=wp.array(dtype=wp.mat33d, ndim=3)
    )),
    
    # 4D kernels
    wp.overload(SO3_Adj_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        out=wp.array(dtype=wp.mat33h, ndim=4)
    )),
    wp.overload(SO3_Adj_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        out=wp.array(dtype=wp.mat33f, ndim=4)
    )),
    wp.overload(SO3_Adj_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        out=wp.array(dtype=wp.mat33d, ndim=4)
    )),
]
