# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
#
# From PyPose SO3_AdjXa.backward:
#   X_grad = -grad_output @ so3_adj(out)  (where so3_adj = skew matrix)
#   a_grad = grad_output @ SO3_Adj(X)     (where SO3_Adj = rotation matrix)
#
# In PyPose these are row vector @ matrix. In Warp with column vectors:
#   X_grad_xyz = -grad_output @ skew(out) = -out × grad_output (cross product)
#   a_grad = grad_output @ R = R^T @ grad_output (for rotation back)
# =============================================================================

def SO3_AdjXa_bwd_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_a: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        R = wp.quat_to_matrix(X[i])
        g = grad_output[i]
        o = out[i]
        
        # grad_a = R^T @ g
        grad_a[i] = wp.transpose(R) @ g
        
        # grad_X_xyz = -g @ skew(o) in row form
        # In column form: -skew(o)^T @ g = skew(o) @ g = o × g
        gx = wp.cross(o, g)
        grad_X[i] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def SO3_AdjXa_bwd_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_a: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j])
        g = grad_output[i, j]
        o = out[i, j]
        
        grad_a[i, j] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i, j] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def SO3_AdjXa_bwd_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_a: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k])
        g = grad_output[i, j, k]
        o = out[i, j, k]
        
        grad_a[i, j, k] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i, j, k] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def SO3_AdjXa_bwd_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_a: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k, l])
        g = grad_output[i, j, k, l]
        o = out[i, j, k, l]
        
        grad_a[i, j, k, l] = wp.transpose(R) @ g
        gx = wp.cross(o, g)
        grad_X[i, j, k, l] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_AdjXa_bwd_kernel_factories = {
    1: SO3_AdjXa_bwd_kernel_1d,
    2: SO3_AdjXa_bwd_kernel_2d,
    3: SO3_AdjXa_bwd_kernel_3d,
    4: SO3_AdjXa_bwd_kernel_4d,
}

_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SO3_AdjXa_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


_TORCH_TO_WP_SCALAR = {
    torch.float16: wp.float16,
    torch.float32: wp.float32,
    torch.float64: wp.float64,
}


# =============================================================================
# Main backward function
# =============================================================================

def SO3_AdjXa_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_AdjXa with respect to inputs.
    
    Args:
        X: Input quaternion of shape (..., 4)
        out: Forward pass output of shape (..., 3)
        grad_output: Gradient w.r.t output of shape (..., 3)
        
    Returns:
        Tuple of (grad_X, grad_a):
            grad_X: shape (..., 4) with w component = 0
            grad_a: shape (..., 3)
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
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    X_wp = wp.from_torch(X, dtype=quat_type)
    out_wp = wp.from_torch(out, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_a_tensor = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_a_wp = wp.from_torch(grad_a_tensor, dtype=vec3_type)
    
    kernel = _get_kernel(ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_a_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_a_tensor = grad_a_tensor.squeeze(0)
    
    return grad_X_tensor, grad_a_tensor
