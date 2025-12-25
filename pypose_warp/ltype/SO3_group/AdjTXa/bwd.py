# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
#
# From PyPose SO3_AdjTXa.backward:
#   a_grad = SO3_AdjXa(X, grad_output) = R @ grad_output
#   X_grad = -a @ so3_adj(a_grad) = -a @ skew(a_grad)
#
# In PyPose these are row vector @ matrix. In Warp with column vectors:
#   a_grad = R @ grad_output (already column form)
#   X_grad_xyz = -a @ skew(a_grad) in row form
#              = -skew(a_grad)^T @ a in column form
#              = skew(a_grad) @ a = a_grad × a
# =============================================================================

def SO3_AdjTXa_bwd_kernel_1d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        a: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_a: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        R = wp.quat_to_matrix(X[i])
        g = grad_output[i]
        av = a[i]
        
        # a_grad = R @ g
        a_grad_vec = R @ g
        grad_a[i] = a_grad_vec
        
        # X_grad_xyz = a_grad × a (cross product)
        gx = wp.cross(a_grad_vec, av)
        grad_X[i] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def SO3_AdjTXa_bwd_kernel_2d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        a: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_a: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j])
        g = grad_output[i, j]
        av = a[i, j]
        
        a_grad_vec = R @ g
        grad_a[i, j] = a_grad_vec
        
        gx = wp.cross(a_grad_vec, av)
        grad_X[i, j] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def SO3_AdjTXa_bwd_kernel_3d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        a: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_a: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k])
        g = grad_output[i, j, k]
        av = a[i, j, k]
        
        a_grad_vec = R @ g
        grad_a[i, j, k] = a_grad_vec
        
        gx = wp.cross(a_grad_vec, av)
        grad_X[i, j, k] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


def SO3_AdjTXa_bwd_kernel_4d(dtype):
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        a: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_a: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        R = wp.quat_to_matrix(X[i, j, k, l])
        g = grad_output[i, j, k, l]
        av = a[i, j, k, l]
        
        a_grad_vec = R @ g
        grad_a[i, j, k, l] = a_grad_vec
        
        gx = wp.cross(a_grad_vec, av)
        grad_X[i, j, k, l] = wp.quaternion(gx[0], gx[1], gx[2], dtype(0.0))
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_AdjTXa_bwd_kernel_factories = {
    1: SO3_AdjTXa_bwd_kernel_1d,
    2: SO3_AdjTXa_bwd_kernel_2d,
    3: SO3_AdjTXa_bwd_kernel_3d,
    4: SO3_AdjTXa_bwd_kernel_4d,
}

_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SO3_AdjTXa_bwd_kernel_factories[ndim]
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

def SO3_AdjTXa_bwd(
    X: torch.Tensor,
    a: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_AdjTXa with respect to inputs.
    
    Args:
        X: Input quaternion of shape (..., 4)
        a: Input Lie algebra element of shape (..., 3)
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
        a = a.unsqueeze(0)
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
    a = a.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    X_wp = wp.from_torch(X, dtype=quat_type)
    a_wp = wp.from_torch(a, dtype=vec3_type)
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
        inputs=[X_wp, a_wp, grad_output_wp, grad_X_wp, grad_a_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_a_tensor = grad_a_tensor.squeeze(0)
    
    return grad_X_tensor, grad_a_tensor

