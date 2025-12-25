# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type


# =============================================================================
# Helper function for computing SO3_Mul backward
#
# From PyPose SO3_Mul.backward:
#   X_grad = (grad_output[..., :3], 0)
#   Y_grad = (grad_output[..., :3] @ SO3_Adj(X), 0)
# where SO3_Adj(X) is the rotation matrix from quaternion X.
# =============================================================================

def compute_mul_grad(dtype):
    @wp.func
    def implement(X: T.Any, g: T.Any) -> T.Any:
        """Compute grad_Y for SO3_Mul backward.
        
        grad_X is just (g, 0), computed separately.
        grad_Y = (g @ R, 0) where R = SO3_Adj(X) = quat_to_matrix(X)
        In Warp column vector form: grad_Y_vec = R^T @ g
        """
        R = wp.quat_to_matrix(X)
        # PyPose: grad = grad_output @ R (row vector @ matrix)
        # Warp: grad = R^T @ g (matrix @ column vector)
        grad_vec = wp.transpose(R) @ g
        return wp.quaternion(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    return implement


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SO3_Mul_bwd_kernel_1d(dtype):
    compute_mul_grad_impl = compute_mul_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_Y: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        g = grad_output[i]
        # grad_X = (g, 0)
        grad_X[i] = wp.quaternion(g[0], g[1], g[2], dtype(0.0))
        # grad_Y = (g @ R, 0)
        grad_Y[i] = compute_mul_grad_impl(X[i], g)
    return implement


def SO3_Mul_bwd_kernel_2d(dtype):
    compute_mul_grad_impl = compute_mul_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_Y: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        g = grad_output[i, j]
        grad_X[i, j] = wp.quaternion(g[0], g[1], g[2], dtype(0.0))
        grad_Y[i, j] = compute_mul_grad_impl(X[i, j], g)
    return implement


def SO3_Mul_bwd_kernel_3d(dtype):
    compute_mul_grad_impl = compute_mul_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_Y: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        g = grad_output[i, j, k]
        grad_X[i, j, k] = wp.quaternion(g[0], g[1], g[2], dtype(0.0))
        grad_Y[i, j, k] = compute_mul_grad_impl(X[i, j, k], g)
    return implement


def SO3_Mul_bwd_kernel_4d(dtype):
    compute_mul_grad_impl = compute_mul_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_Y: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        g = grad_output[i, j, k, l]
        grad_X[i, j, k, l] = wp.quaternion(g[0], g[1], g[2], dtype(0.0))
        grad_Y[i, j, k, l] = compute_mul_grad_impl(X[i, j, k, l], g)
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_Mul_bwd_kernel_factories = {
    1: SO3_Mul_bwd_kernel_1d,
    2: SO3_Mul_bwd_kernel_2d,
    3: SO3_Mul_bwd_kernel_3d,
    4: SO3_Mul_bwd_kernel_4d,
}

_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SO3_Mul_bwd_kernel_factories[ndim]
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

def SO3_Mul_bwd(
    X: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients of SO3_Mul with respect to input quaternions.
    
    Args:
        X: First input quaternion of shape (..., 4)
        grad_output: Gradient w.r.t output of shape (..., 4)
        
    Returns:
        Tuple of (grad_X, grad_Y), each of shape (..., 4)
    """
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        X = X.unsqueeze(0)
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
    grad_output = grad_output.detach().contiguous()
    
    # Extract xyz components of grad_output for the backward computation
    grad_output_xyz = grad_output[..., :3].contiguous()
    
    X_wp = wp.from_torch(X, dtype=quat_type)
    grad_output_wp = wp.from_torch(grad_output_xyz, dtype=vec3_type)
    
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_Y_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_Y_wp = wp.from_torch(grad_Y_tensor, dtype=quat_type)
    
    kernel = _get_kernel(ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, grad_output_wp, grad_X_wp, grad_Y_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_Y_tensor = grad_Y_tensor.squeeze(0)
    
    return grad_X_tensor, grad_Y_tensor

