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
#
# SO3_AdjTXa forward:
#   out = SO3_Adj(X^{-1}) @ a = R^T @ a
# where R = quat_to_matrix(X) is the 3x3 rotation matrix.
# =============================================================================

@wp.kernel(enable_backward=False)
def SO3_AdjTXa_fwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    a: wp.array(dtype=T.Any, ndim=1),
    out: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    R = wp.quat_to_matrix(X[i])
    # R^T @ a (transpose of rotation matrix applied to a)
    out[i] = wp.transpose(R) @ a[i]


@wp.kernel(enable_backward=False)
def SO3_AdjTXa_fwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    a: wp.array(dtype=T.Any, ndim=2),
    out: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid()  # type: ignore
    R = wp.quat_to_matrix(X[i, j])
    out[i, j] = wp.transpose(R) @ a[i, j]


@wp.kernel(enable_backward=False)
def SO3_AdjTXa_fwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    a: wp.array(dtype=T.Any, ndim=3),
    out: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid()  # type: ignore
    R = wp.quat_to_matrix(X[i, j, k])
    out[i, j, k] = wp.transpose(R) @ a[i, j, k]


@wp.kernel(enable_backward=False)
def SO3_AdjTXa_fwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    a: wp.array(dtype=T.Any, ndim=4),
    out: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid()  # type: ignore
    R = wp.quat_to_matrix(X[i, j, k, l])
    out[i, j, k, l] = wp.transpose(R) @ a[i, j, k, l]


# =============================================================================
# Kernel selection map
# =============================================================================

_SO3_AdjTXa_kernels = {
    1: SO3_AdjTXa_fwd_kernel_1d,
    2: SO3_AdjTXa_fwd_kernel_2d,
    3: SO3_AdjTXa_fwd_kernel_3d,
    4: SO3_AdjTXa_fwd_kernel_4d,
}


# =============================================================================
# Main forward function with arbitrary dimension support
# =============================================================================

def SO3_AdjTXa_fwd(X: pp.LieTensor, a: torch.Tensor) -> torch.Tensor:
    """
    Compute the adjoint transpose action of SO3 element X on Lie algebra element a.
    
    This computes: out = SO3_Adj(X^{-1}) @ a = R^T @ a
    where R is the 3x3 rotation matrix corresponding to quaternion X.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        a: Tensor of shape (..., 3) - Lie algebra element (so3)
        
    Returns:
        Tensor of shape (broadcast(...), 3) - transformed Lie algebra element
    """
    X_tensor = X.tensor()
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]
    a_batch_shape = a.shape[:-1]
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, a_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X.shape} and {a.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        a = a.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape (creates stride-0 views, no data copy)
    X_expanded = X_tensor.expand(*out_batch_shape, 4)
    a_expanded = a.expand(*out_batch_shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    
    # Convert to warp arrays (preserves strides including stride-0)
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    a_wp = wp.from_torch(a_expanded, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Launch kernel with multi-dimensional grid
    wp.launch(
        kernel=_SO3_AdjTXa_kernels[ndim],
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, a_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# =============================================================================

__SO3_AdjTXa_fwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_AdjTXa_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        a=wp.array(dtype=wp.vec3h, ndim=1),
        out=wp.array(dtype=wp.vec3h, ndim=1)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        a=wp.array(dtype=wp.vec3f, ndim=1),
        out=wp.array(dtype=wp.vec3f, ndim=1)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        a=wp.array(dtype=wp.vec3d, ndim=1),
        out=wp.array(dtype=wp.vec3d, ndim=1)
    )),
    
    # 2D kernels
    wp.overload(SO3_AdjTXa_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        a=wp.array(dtype=wp.vec3h, ndim=2),
        out=wp.array(dtype=wp.vec3h, ndim=2)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        a=wp.array(dtype=wp.vec3f, ndim=2),
        out=wp.array(dtype=wp.vec3f, ndim=2)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        a=wp.array(dtype=wp.vec3d, ndim=2),
        out=wp.array(dtype=wp.vec3d, ndim=2)
    )),
    
    # 3D kernels
    wp.overload(SO3_AdjTXa_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        a=wp.array(dtype=wp.vec3h, ndim=3),
        out=wp.array(dtype=wp.vec3h, ndim=3)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        a=wp.array(dtype=wp.vec3f, ndim=3),
        out=wp.array(dtype=wp.vec3f, ndim=3)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        a=wp.array(dtype=wp.vec3d, ndim=3),
        out=wp.array(dtype=wp.vec3d, ndim=3)
    )),
    
    # 4D kernels
    wp.overload(SO3_AdjTXa_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        a=wp.array(dtype=wp.vec3h, ndim=4),
        out=wp.array(dtype=wp.vec3h, ndim=4)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        a=wp.array(dtype=wp.vec3f, ndim=4),
        out=wp.array(dtype=wp.vec3f, ndim=4)
    )),
    wp.overload(SO3_AdjTXa_fwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        a=wp.array(dtype=wp.vec3d, ndim=4),
        out=wp.array(dtype=wp.vec3d, ndim=4)
    )),
]

