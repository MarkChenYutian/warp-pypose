# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type


# =============================================================================
# Helper function for computing SO3_Inv backward
#
# From PyPose SO3_Inv.backward:
#   X_grad = -(grad_output[..., :-1] @ SO3_Adj(Y))
# where Y is the inverse quaternion and SO3_Adj computes the rotation matrix.
# =============================================================================

def compute_inv_grad(dtype):
    @wp.func
    def implement(Y: T.Any, g: T.Any) -> T.Any:
        """Compute grad_X for SO3_Inv backward."""
        # SO3_Adj(Y) is the rotation matrix from quaternion Y
        R = wp.quat_to_matrix(Y)
        # PyPose: grad = -(grad_output[..., :3] @ R) (row vector @ matrix)
        # Warp: grad = -(R^T @ g) (matrix @ column vector)
        grad_vec = wp.transpose(R) @ g
        grad_vec = -grad_vec
        return wp.quaternion(grad_vec[0], grad_vec[1], grad_vec[2], dtype(0.0))
    return implement


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SO3_Inv_bwd_kernel_1d(dtype):
    compute_inv_grad_impl = compute_inv_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_inv_grad_impl(out[i], grad_output[i])
    return implement


def SO3_Inv_bwd_kernel_2d(dtype):
    compute_inv_grad_impl = compute_inv_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_inv_grad_impl(out[i, j], grad_output[i, j])
    return implement


def SO3_Inv_bwd_kernel_3d(dtype):
    compute_inv_grad_impl = compute_inv_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_inv_grad_impl(out[i, j, k], grad_output[i, j, k])
    return implement


def SO3_Inv_bwd_kernel_4d(dtype):
    compute_inv_grad_impl = compute_inv_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_inv_grad_impl(out[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_Inv_bwd_kernel_factories = {
    1: SO3_Inv_bwd_kernel_1d,
    2: SO3_Inv_bwd_kernel_2d,
    3: SO3_Inv_bwd_kernel_3d,
    4: SO3_Inv_bwd_kernel_4d,
}

_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SO3_Inv_bwd_kernel_factories[ndim]
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

def SO3_Inv_bwd(
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of SO3_Inv with respect to input quaternion.
    
    Args:
        out: Forward output (inverse quaternion) of shape (..., 4)
        grad_output: Gradient w.r.t output of shape (..., 4)
        
    Returns:
        Gradient w.r.t input quaternion of shape (..., 4)
    """
    batch_shape = out.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    dtype = out.dtype
    device = out.device
    
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Extract xyz components of grad_output for the backward computation
    grad_output_xyz = grad_output[..., :3].contiguous()
    
    out_wp = wp.from_torch(out, dtype=quat_type)
    grad_output_wp = wp.from_torch(grad_output_xyz, dtype=vec3_type)
    
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    
    kernel = _get_kernel(ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=out_wp.device,
        inputs=[out_wp, grad_output_wp, grad_X_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
    
    return grad_X_tensor

