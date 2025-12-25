# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type


# =============================================================================
# Kernels for different batch dimensions (1D to 4D)
# Warp requires fixed ndim at compile time, so we need separate kernels.
# =============================================================================

@wp.kernel(enable_backward=False)
def SO3_Mul_fwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    Y: wp.array(dtype=T.Any, ndim=1),
    out: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    out[i] = wp.mul(X[i], Y[i])


@wp.kernel(enable_backward=False)
def SO3_Mul_fwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    Y: wp.array(dtype=T.Any, ndim=2),
    out: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid()  # type: ignore
    out[i, j] = wp.mul(X[i, j], Y[i, j])


@wp.kernel(enable_backward=False)
def SO3_Mul_fwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    Y: wp.array(dtype=T.Any, ndim=3),
    out: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid()  # type: ignore
    out[i, j, k] = wp.mul(X[i, j, k], Y[i, j, k])


@wp.kernel(enable_backward=False)
def SO3_Mul_fwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    Y: wp.array(dtype=T.Any, ndim=4),
    out: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid()  # type: ignore
    out[i, j, k, l] = wp.mul(X[i, j, k, l], Y[i, j, k, l])


# =============================================================================
# Kernel selection map
# =============================================================================

_SO3_Mul_kernels = {
    1: SO3_Mul_fwd_kernel_1d,
    2: SO3_Mul_fwd_kernel_2d,
    3: SO3_Mul_fwd_kernel_3d,
    4: SO3_Mul_fwd_kernel_4d,
}


# =============================================================================
# Main forward function with arbitrary dimension support
# =============================================================================

def SO3_Mul_fwd(X: pp.LieTensor, Y: pp.LieTensor) -> pp.LieTensor:
    """
    Compute SO3 group multiplication X * Y (quaternion multiplication).
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        Y: SO3 LieTensor of shape (..., 4) - quaternion representation
        
    Returns:
        SO3 LieTensor of shape (broadcast(...), 4) - product quaternion
    """
    X_tensor = X.tensor()
    Y_tensor = Y.tensor()
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]
    Y_batch_shape = Y_tensor.shape[:-1]
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, Y_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X.shape} and {Y.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        Y_tensor = Y_tensor.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape (creates stride-0 views, no data copy)
    X_expanded = X_tensor.expand(*out_batch_shape, 4)
    Y_expanded = Y_tensor.expand(*out_batch_shape, 4)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    
    # Convert to warp arrays (preserves strides including stride-0)
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    Y_wp = wp.from_torch(Y_expanded, dtype=quat_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 4), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=quat_type)
    
    # Launch kernel with multi-dimensional grid
    wp.launch(
        kernel=_SO3_Mul_kernels[ndim],
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, Y_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return pp.LieTensor(out_tensor, ltype=pp.SO3_type)


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# =============================================================================

__SO3_Mul_fwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_Mul_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        Y=wp.array(dtype=wp.quath, ndim=1),
        out=wp.array(dtype=wp.quath, ndim=1)
    )),
    wp.overload(SO3_Mul_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        Y=wp.array(dtype=wp.quatf, ndim=1),
        out=wp.array(dtype=wp.quatf, ndim=1)
    )),
    wp.overload(SO3_Mul_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        Y=wp.array(dtype=wp.quatd, ndim=1),
        out=wp.array(dtype=wp.quatd, ndim=1)
    )),
    
    # 2D kernels
    wp.overload(SO3_Mul_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        Y=wp.array(dtype=wp.quath, ndim=2),
        out=wp.array(dtype=wp.quath, ndim=2)
    )),
    wp.overload(SO3_Mul_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        Y=wp.array(dtype=wp.quatf, ndim=2),
        out=wp.array(dtype=wp.quatf, ndim=2)
    )),
    wp.overload(SO3_Mul_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        Y=wp.array(dtype=wp.quatd, ndim=2),
        out=wp.array(dtype=wp.quatd, ndim=2)
    )),
    
    # 3D kernels
    wp.overload(SO3_Mul_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        Y=wp.array(dtype=wp.quath, ndim=3),
        out=wp.array(dtype=wp.quath, ndim=3)
    )),
    wp.overload(SO3_Mul_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        Y=wp.array(dtype=wp.quatf, ndim=3),
        out=wp.array(dtype=wp.quatf, ndim=3)
    )),
    wp.overload(SO3_Mul_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        Y=wp.array(dtype=wp.quatd, ndim=3),
        out=wp.array(dtype=wp.quatd, ndim=3)
    )),
    
    # 4D kernels
    wp.overload(SO3_Mul_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        Y=wp.array(dtype=wp.quath, ndim=4),
        out=wp.array(dtype=wp.quath, ndim=4)
    )),
    wp.overload(SO3_Mul_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        Y=wp.array(dtype=wp.quatf, ndim=4),
        out=wp.array(dtype=wp.quatf, ndim=4)
    )),
    wp.overload(SO3_Mul_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        Y=wp.array(dtype=wp.quatd, ndim=4),
        out=wp.array(dtype=wp.quatd, ndim=4)
    )),
]

