# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type, wp_vec4_type


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
# Backward kernel factories for 1D to 4D batch dimensions
#
# From PyPose SO3_Act4.backward:
#   X_grad = grad_output @ SO3_Act4_Jacobian(out)
#   p_grad = grad_output @ SO3_Matrix4x4(X)
#
# Where:
#   SO3_Act4_Jacobian(out) is 4x3 with top 3x3 = skew(-out[:3]), bottom row = 0
#   SO3_Matrix4x4(X) is 4x4 with top-left 3x3 = R, (3,3) = 1
#
# In row form (PyPose):
#   X_grad = grad_output[:3] @ skew(-out[:3]) (only first 3 components contribute)
#   p_grad[:3] = grad_output[:3] @ R
#   p_grad[3] = grad_output[3]
#
# In column form (Warp):
#   X_grad_xyz = skew(-out[:3])^T @ grad[:3] = -skew(-out[:3]) @ grad[:3] 
#              = skew(out[:3]) @ grad[:3] = out[:3] × grad[:3]
#   p_grad[:3] = R^T @ grad[:3]
#   p_grad[3] = grad[3]
# =============================================================================

def SO3_Act4_bwd_kernel_1d(dtype, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        R = wp.quat_to_matrix(X[i])
        o = out[i]
        g = grad_output[i]
        
        # Extract first 3 components
        o3 = extract_func(o)
        g3 = extract_func(g)
        
        # X_grad_xyz = out[:3] × grad[:3]
        gx = wp.cross(o3, g3)
        grad_X[i] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        # p_grad[:3] = R^T @ grad[:3], p_grad[3] = grad[3]
        pg3 = wp.transpose(R) @ g3
        grad_p[i] = vec4_type(pg3[0], pg3[1], pg3[2], g[3])
    return implement


def SO3_Act4_bwd_kernel_2d(dtype, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j])
        o = out[i, j]
        g = grad_output[i, j]
        
        o3 = extract_func(o)
        g3 = extract_func(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i, j] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.transpose(R) @ g3
        grad_p[i, j] = vec4_type(pg3[0], pg3[1], pg3[2], g[3])
    return implement


def SO3_Act4_bwd_kernel_3d(dtype, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k])
        o = out[i, j, k]
        g = grad_output[i, j, k]
        
        o3 = extract_func(o)
        g3 = extract_func(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i, j, k] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.transpose(R) @ g3
        grad_p[i, j, k] = vec4_type(pg3[0], pg3[1], pg3[2], g[3])
    return implement


def SO3_Act4_bwd_kernel_4d(dtype, vec4_type, extract_func):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k, l])
        o = out[i, j, k, l]
        g = grad_output[i, j, k, l]
        
        o3 = extract_func(o)
        g3 = extract_func(g)
        
        gx = wp.cross(o3, g3)
        grad_X[i, j, k, l] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
        
        pg3 = wp.transpose(R) @ g3
        grad_p[i, j, k, l] = vec4_type(pg3[0], pg3[1], pg3[2], g[3])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_Act4_bwd_kernel_factories = {
    1: SO3_Act4_bwd_kernel_1d,
    2: SO3_Act4_bwd_kernel_2d,
    3: SO3_Act4_bwd_kernel_3d,
    4: SO3_Act4_bwd_kernel_4d,
}

_DTYPE_CONFIGS = {
    torch.float16: (wp.float16, wp.vec4h, extract_vec3h),
    torch.float32: (wp.float32, wp.vec4f, extract_vec3f),
    torch.float64: (wp.float64, wp.vec4d, extract_vec3d),
}

_kernel_cache: dict[tuple[int, torch.dtype], T.Any] = {}


def _get_kernel(ndim: int, dtype: torch.dtype):
    key = (ndim, dtype)
    if key not in _kernel_cache:
        scalar_type, vec4_type, extract_func = _DTYPE_CONFIGS[dtype]
        factory = _SO3_Act4_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(scalar_type, vec4_type, extract_func)
    return _kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def SO3_Act4_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_Act4 with respect to inputs.
    
    Args:
        X: Input quaternion of shape (..., 4)
        out: Forward pass output of shape (..., 4)
        grad_output: Gradient w.r.t output of shape (..., 4)
        
    Returns:
        Tuple of (grad_X, grad_p):
            grad_X: shape (..., 4) with w component = 0
            grad_p: shape (..., 4)
    """
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        X = X.unsqueeze(0)
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    dtype = X.dtype
    device = X.device
    
    quat_type = wp_quat_type(dtype)
    vec4_type = wp_vec4_type(dtype)
    
    # Detach and ensure contiguous
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    X_wp = wp.from_torch(X, dtype=quat_type)
    out_wp = wp.from_torch(out, dtype=vec4_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec4_type)
    
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec4_type)
    
    kernel = _get_kernel(ndim, dtype)
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_p_tensor = grad_p_tensor.squeeze(0)
    
    return grad_X_tensor, grad_p_tensor
