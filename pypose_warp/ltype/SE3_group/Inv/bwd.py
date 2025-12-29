# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    wp_transform,
)


# =============================================================================
# SE3_Inv Backward Pass
#
# Forward: Y = inv(X) where Y = (t_inv, q_inv)
#
# Backward (from PyPose):
#   X_grad[:6] = -(grad_output[:6] @ SE3_Adj(Y))
#   X_grad[6] = 0
#
# Where SE3_Adj(Y) is a 6x6 matrix:
#   Adj[:3, :3] = R        (rotation matrix from q_inv)
#   Adj[:3, 3:] = skew(t) @ R
#   Adj[3:, :3] = 0
#   Adj[3:, 3:] = R
#
# The gradient is: X_grad = -grad @ Adj_Y
# which expands to:
#   X_grad[:3] = -(grad[:3] @ R + grad[3:6] @ 0) = -grad[:3] @ R
#   X_grad[3:6] = -(grad[:3] @ skew(t)@R + grad[3:6] @ R)
#              = -(cross(t, grad[:3]) @ R + grad[3:6] @ R)
#              = -R^T @ (cross(t, grad[:3]) + grad[3:6])
#   X_grad[6] = 0
#
# Actually in column form (Warp uses column vectors):
#   X_grad[:3] = -(R^T @ grad[:3])
#   X_grad[3:6] = -(R^T @ (skew(t)^T @ grad[:3] + grad[3:6]))
#              = -(R^T @ (cross(grad[:3], t) + grad[3:6]))
# =============================================================================


def _make_inv_grad_func(dtype):
    """
    Factory function to create dtype-specific gradient computation function.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        compute_se3_inv_grad warp function
    """
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_se3_inv_grad(Y: T.Any, grad: T.Any) -> T.Any:
        """
        Compute gradient of SE3 inverse.
        
        Y is the output (inverse), grad is the gradient w.r.t. Y.
        Returns gradient w.r.t. X.
        
        Formula: X_grad = -grad @ SE3_Adj(Y) (in row form)
        In column form:
            X_grad[:3] = -(R^T @ grad_t)
            X_grad[3:6] = -(R^T @ (cross(grad_t, t) + grad_r))
        """
        # Extract components from Y (the inverse)
        t = wp.transform_get_translation(Y)
        q = wp.transform_get_rotation(Y)
        R = wp.quat_to_matrix(q)
        RT = wp.transpose(R)
        
        # Extract gradient components
        grad_t = wp.transform_get_translation(grad)  # First 3 components
        grad_r = vec3_ctor(grad[3], grad[4], grad[5])  # Components 3,4,5 (rotation part)
        
        # Compute X_grad[:3] = -(R^T @ grad_t)
        X_grad_t = -(RT @ grad_t)
        
        # Compute X_grad[3:6] = -(R^T @ (cross(grad_t, t) + grad_r))
        # Note: skew(t)^T @ grad_t = -cross(t, grad_t) = cross(grad_t, t)
        X_grad_r = -(RT @ (wp.cross(grad_t, t) + grad_r))
        
        # Return as transform with quaternion (xyz components of grad_r, 0)
        return transform_ctor(X_grad_t, quat_ctor(X_grad_r[0], X_grad_r[1], X_grad_r[2], dtype(0.0)))

    return compute_se3_inv_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_Inv_bwd_kernel_1d(dtype):
    compute_grad = _make_inv_grad_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        Y: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad(Y[i], grad_output[i])
    return implement


def SE3_Inv_bwd_kernel_2d(dtype):
    compute_grad = _make_inv_grad_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        Y: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad(Y[i, j], grad_output[i, j])
    return implement


def SE3_Inv_bwd_kernel_3d(dtype):
    compute_grad = _make_inv_grad_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        Y: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad(Y[i, j, k], grad_output[i, j, k])
    return implement


def SE3_Inv_bwd_kernel_4d(dtype):
    compute_grad = _make_inv_grad_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        Y: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad(Y[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Inv_bwd_kernel_factories = {
    1: SE3_Inv_bwd_kernel_1d,
    2: SE3_Inv_bwd_kernel_2d,
    3: SE3_Inv_bwd_kernel_3d,
    4: SE3_Inv_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Inv_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def SE3_Inv_bwd(
    Y: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for SE3_Inv.
    
    Args:
        Y: Output from forward pass (the inverse), shape (..., 7)
        grad_output: Gradient w.r.t. output, shape (..., 7)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 7)
    """
    batch_shape = Y.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
        Y = Y.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    dtype = Y.dtype
    device = Y.device
    transform_type = wp_transform(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous for warp conversion
    Y = Y.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    Y_wp = wp.from_torch(Y, dtype=transform_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=transform_type)
    
    # Allocate output gradient
    grad_X_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=Y_wp.device,
        inputs=[Y_wp, grad_output_wp, grad_X_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
    
    return grad_X_tensor

