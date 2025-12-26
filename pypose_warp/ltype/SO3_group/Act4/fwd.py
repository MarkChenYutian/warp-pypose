# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec4_type


# =============================================================================
# Helper to extract first 3 components from vec4 as vec3
# =============================================================================

@wp.func
def extract_vec3h(v: wp.vec4h) -> wp.vec3h:
    return wp.vec3h(v[0], v[1], v[2])

@wp.func
def extract_vec3f(v: wp.vec4f) -> wp.vec3f:
    return wp.vec3f(v[0], v[1], v[2])

@wp.func
def extract_vec3d(v: wp.vec4d) -> wp.vec3d:
    return wp.vec3d(v[0], v[1], v[2])


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# Warp requires fixed ndim at compile time, so we need separate kernels.
#
# SO3_Act4 forward:
#   out = [SO3_Act(X, p[:3]), p[3]]
# Rotates the first 3 components, keeps the 4th unchanged.
# =============================================================================

def SO3_Act4_fwd_kernel_1d(vec3_type, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        q = X[i]
        pv = p[i]
        # Rotate first 3 components using quaternion rotation
        p3 = extract_func(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i] = vec4_type(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


def SO3_Act4_fwd_kernel_2d(vec3_type, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        q = X[i, j]
        pv = p[i, j]
        p3 = extract_func(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i, j] = vec4_type(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


def SO3_Act4_fwd_kernel_3d(vec3_type, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        q = X[i, j, k]
        pv = p[i, j, k]
        p3 = extract_func(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i, j, k] = vec4_type(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


def SO3_Act4_fwd_kernel_4d(vec3_type, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        q = X[i, j, k, l]
        pv = p[i, j, k, l]
        p3 = extract_func(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i, j, k, l] = vec4_type(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


# =============================================================================
# Kernel selection map and caching
# =============================================================================

_SO3_Act4_kernel_factories = {
    1: SO3_Act4_fwd_kernel_1d,
    2: SO3_Act4_fwd_kernel_2d,
    3: SO3_Act4_fwd_kernel_3d,
    4: SO3_Act4_fwd_kernel_4d,
}

_DTYPE_CONFIGS = {
    torch.float16: (wp.vec3h, wp.vec4h, extract_vec3h),
    torch.float32: (wp.vec3f, wp.vec4f, extract_vec3f),
    torch.float64: (wp.vec3d, wp.vec4d, extract_vec3d),
}

_kernel_cache: dict[tuple[int, torch.dtype], T.Any] = {}


def _get_kernel(ndim: int, dtype: torch.dtype):
    key = (ndim, dtype)
    if key not in _kernel_cache:
        vec3_type, vec4_type, extract_func = _DTYPE_CONFIGS[dtype]
        factory = _SO3_Act4_kernel_factories[ndim]
        _kernel_cache[key] = factory(vec3_type, vec4_type, extract_func)
    return _kernel_cache[key]


# =============================================================================
# Main forward function with arbitrary dimension support
# =============================================================================

def SO3_Act4_fwd(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
    """
    Apply SO3 rotation to 4D homogeneous points (rotate first 3 components).
    
    This computes: out = [R @ p[:3], p[3]]
    where R is the 3x3 rotation matrix corresponding to quaternion X.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        p: Tensor of shape (..., 4) - 4D homogeneous points
        
    Returns:
        Tensor of shape (broadcast(...), 4) - rotated 4D points
    """
    X_tensor = X.tensor()
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]
    p_batch_shape = p.shape[:-1]
    
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
    p_expanded = p.expand(*out_batch_shape, 4)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec4_type = wp_vec4_type(dtype)
    
    # Convert to warp arrays (preserves strides including stride-0)
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec4_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 4), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec4_type)
    
    # Launch kernel with multi-dimensional grid
    kernel = _get_kernel(ndim, dtype)
    wp.launch(
        kernel=kernel,
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor



