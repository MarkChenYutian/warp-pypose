# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type


# =============================================================================
# Kernels for different batch dimensions (1D to 4D)
# Warp requires fixed ndim at compile time, so we need separate kernels.
# Each kernel uses multi-dimensional indexing which respects strides (including
# stride-0 for broadcast dimensions).
# =============================================================================

@wp.kernel(enable_backward=False)
def SO3_Act_fwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    p: wp.array(dtype=T.Any, ndim=1),
    out: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    out[i] = wp.quat_rotate(X[i], p[i])


@wp.kernel(enable_backward=False)
def SO3_Act_fwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    p: wp.array(dtype=T.Any, ndim=2),
    out: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid() #type: ignore
    out[i, j] = wp.quat_rotate(X[i, j], p[i, j])


@wp.kernel(enable_backward=False)
def SO3_Act_fwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    p: wp.array(dtype=T.Any, ndim=3),
    out: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid() #type: ignore
    out[i, j, k] = wp.quat_rotate(X[i, j, k], p[i, j, k])


@wp.kernel(enable_backward=False)
def SO3_Act_fwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    p: wp.array(dtype=T.Any, ndim=4),
    out: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid() #type: ignore
    out[i, j, k, l] = wp.quat_rotate(X[i, j, k, l], p[i, j, k, l])


# =============================================================================
# Kernel selection map: (ndim) -> kernel function
# =============================================================================

_SO3_Act_kernels = {
    1: SO3_Act_fwd_kernel_1d,
    2: SO3_Act_fwd_kernel_2d,
    3: SO3_Act_fwd_kernel_3d,
    4: SO3_Act_fwd_kernel_4d,
}


# =============================================================================
# Main forward function with arbitrary dimension support
# =============================================================================

def SO3_Act_fwd(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
    """
    Apply SO3 rotation X to 3D points p.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        p: Tensor of shape (..., 3) - 3D points
        
    Returns:
        Rotated points of shape (broadcast(...), 3)
    """
    X_tensor = X.tensor()
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]  # (...,) from (..., 4)
    p_batch_shape = p.shape[:-1]         # (...,) from (..., 3)
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X.shape} and {p.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        p = p.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape (creates stride-0 views, no data copy)
    X_expanded = X_tensor.expand(*out_batch_shape, 4)
    p_expanded = p.expand(*out_batch_shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    
    # Convert to warp arrays (preserves strides including stride-0)
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Launch kernel with multi-dimensional grid
    wp.launch(
        kernel=_SO3_Act_kernels[ndim],
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# This ensures warp compiles specialized versions for each type combination.
# =============================================================================

__SO3_Act_fwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_Act_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        p=wp.array(dtype=wp.vec3h, ndim=1),
        out=wp.array(dtype=wp.vec3h, ndim=1)
    )),
    wp.overload(SO3_Act_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        p=wp.array(dtype=wp.vec3f, ndim=1),
        out=wp.array(dtype=wp.vec3f, ndim=1)
    )),
    wp.overload(SO3_Act_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        p=wp.array(dtype=wp.vec3d, ndim=1),
        out=wp.array(dtype=wp.vec3d, ndim=1)
    )),
    
    # 2D kernels
    wp.overload(SO3_Act_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        p=wp.array(dtype=wp.vec3h, ndim=2),
        out=wp.array(dtype=wp.vec3h, ndim=2)
    )),
    wp.overload(SO3_Act_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        p=wp.array(dtype=wp.vec3f, ndim=2),
        out=wp.array(dtype=wp.vec3f, ndim=2)
    )),
    wp.overload(SO3_Act_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        p=wp.array(dtype=wp.vec3d, ndim=2),
        out=wp.array(dtype=wp.vec3d, ndim=2)
    )),
    
    # 3D kernels
    wp.overload(SO3_Act_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        p=wp.array(dtype=wp.vec3h, ndim=3),
        out=wp.array(dtype=wp.vec3h, ndim=3)
    )),
    wp.overload(SO3_Act_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        p=wp.array(dtype=wp.vec3f, ndim=3),
        out=wp.array(dtype=wp.vec3f, ndim=3)
    )),
    wp.overload(SO3_Act_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        p=wp.array(dtype=wp.vec3d, ndim=3),
        out=wp.array(dtype=wp.vec3d, ndim=3)
    )),
    
    # 4D kernels
    wp.overload(SO3_Act_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        p=wp.array(dtype=wp.vec3h, ndim=4),
        out=wp.array(dtype=wp.vec3h, ndim=4)
    )),
    wp.overload(SO3_Act_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        p=wp.array(dtype=wp.vec3f, ndim=4),
        out=wp.array(dtype=wp.vec3f, ndim=4)
    )),
    wp.overload(SO3_Act_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        p=wp.array(dtype=wp.vec3d, ndim=4),
        out=wp.array(dtype=wp.vec3d, ndim=4)
    )),
]
