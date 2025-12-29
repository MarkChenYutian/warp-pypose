# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_VEC4,
    wp_vec4,
    wp_transform,
)


# =============================================================================
# SE3_Act4 Forward Pass
#
# SE3 element X has shape (..., 7): [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# where t is translation (3D) and q is quaternion (4D)
#
# Action on 4D homogeneous point p = [p_x, p_y, p_z, p_w]:
#   out[:3] = R @ p[:3] + t * p[3]  (rotation + scaled translation)
#   out[3] = p[3]                    (preserve homogeneous coordinate)
#
# This is equivalent to multiplying by the SE3 4x4 matrix:
#   [R | t] [p[:3]]   [R @ p[:3] + t * p[3]]
#   [0 | 1] [p[3] ] = [p[3]                ]
# =============================================================================


def _make_se3_act4_point(dtype):
    """
    Factory function to create dtype-specific SE3_Act4 function.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        se3_act4_point warp function
    """
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.func
    def se3_act4_point(X: T.Any, p: T.Any) -> T.Any:
        """Apply SE3 transform to a 4D homogeneous point."""
        # Extract translation and quaternion from transform
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        
        # Extract 3D point and homogeneous coordinate
        p3 = vec3_ctor(p[0], p[1], p[2])
        pw = p[3]
        
        # Rotate 3D point and add scaled translation
        rotated = wp.quat_rotate(q, p3)
        out3 = rotated + t * pw
        
        # Return as vec4 with preserved homogeneous coordinate
        return vec4_ctor(out3[0], out3[1], out3[2], pw)
    
    return se3_act4_point


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def SE3_Act4_fwd_kernel_1d(dtype):
    se3_act4_point = _make_se3_act4_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_act4_point(X[i], p[i])
    return implement


def SE3_Act4_fwd_kernel_2d(dtype):
    se3_act4_point = _make_se3_act4_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_act4_point(X[i, j], p[i, j])
    return implement


def SE3_Act4_fwd_kernel_3d(dtype):
    se3_act4_point = _make_se3_act4_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_act4_point(X[i, j, k], p[i, j, k])
    return implement


def SE3_Act4_fwd_kernel_4d(dtype):
    se3_act4_point = _make_se3_act4_point(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_act4_point(X[i, j, k, l], p[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Act4_fwd_kernel_factories = {
    1: SE3_Act4_fwd_kernel_1d,
    2: SE3_Act4_fwd_kernel_2d,
    3: SE3_Act4_fwd_kernel_3d,
    4: SE3_Act4_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Act4_fwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# =============================================================================
# Main forward function
# =============================================================================

def SE3_Act4_fwd(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
    """
    Apply SE3 transformation X to 4D homogeneous points p.
    
    SE3 action on homogeneous coordinates:
        out[:3] = R @ p[:3] + t * p[3]
        out[3] = p[3]
    
    This is equivalent to: SE3_matrix @ p (4x4 matrix multiply)
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        p: Tensor of shape (..., 4) - 4D homogeneous points
        
    Returns:
        Transformed points of shape (broadcast(...), 4)
    """
    X_tensor = X.tensor()
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]  # (...,) from (..., 7)
    p_batch_shape = p.shape[:-1]         # (...,) from (..., 4)
    
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
    X_expanded = X_tensor.expand(*out_batch_shape, 7)
    p_expanded = p.expand(*out_batch_shape, 4)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform(dtype)
    vec4_type = wp_vec4(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=transform_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec4_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 4), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec4_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor

