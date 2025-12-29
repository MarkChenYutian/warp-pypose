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
    KernelRegistry,
    prepare_batch_broadcast,
    finalize_output,
    wp_vec4,
    wp_quat,
)


# =============================================================================
# Helper to extract first 3 components from vec4 as vec3
# =============================================================================

def _make_extract_vec3(dtype):
    vec3_ctor = DTYPE_TO_VEC3[dtype]

    @wp.func
    def extract_vec3(v: T.Any) -> T.Any:
        return vec3_ctor(v[0], v[1], v[2])
    return extract_vec3


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
#
# SO3_Act4 forward:
#   out = [SO3_Act(X, p[:3]), p[3]]
# Rotates the first 3 components, keeps the 4th unchanged.
# =============================================================================

def _make_kernel_1d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        q = X[i]
        pv = p[i]
        p3 = extract_vec3(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i] = vec4_ctor(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


def _make_kernel_2d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        q = X[i, j]
        pv = p[i, j]
        p3 = extract_vec3(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i, j] = vec4_ctor(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


def _make_kernel_3d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        q = X[i, j, k]
        pv = p[i, j, k]
        p3 = extract_vec3(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i, j, k] = vec4_ctor(rotated[0], rotated[1], rotated[2], pv[3])
    return implement


def _make_kernel_4d(dtype):
    extract_vec3 = _make_extract_vec3(dtype)
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        q = X[i, j, k, l]
        pv = p[i, j, k, l]
        p3 = extract_vec3(pv)
        rotated = wp.quat_rotate(q, p3)
        out[i, j, k, l] = vec4_ctor(rotated[0], rotated[1], rotated[2], pv[3])
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
    
    # Prepare batch dimensions with broadcasting
    X_tensor, p, batch_info = prepare_batch_broadcast(X_tensor, p)
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*batch_info.shape, 4)
    p_expanded = p.expand(*batch_info.shape, 4)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat(dtype)
    vec4_type = wp_vec4(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec4_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec4_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
