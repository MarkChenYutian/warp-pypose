# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_transform_type


# =============================================================================
# SE3_Mul Backward Pass
#
# Forward: out = X @ Y
#   t_out = t_X + R_X @ t_Y
#   q_out = q_X * q_Y
#
# Backward (from PyPose SE3_Mul.backward):
#   X_grad[:6] = grad_output[:6]
#   X_grad[6] = 0
#   
#   Y_grad[:6] = grad_output[:6] @ SE3_Adj(X)
#   Y_grad[6] = 0
#
# Where SE3_Adj(X) is the adjoint representation (6x6 matrix):
#   Adj[:3, :3] = R       (rotation matrix from quaternion)
#   Adj[:3, 3:] = skew(t) @ R
#   Adj[3:, :3] = 0
#   Adj[3:, 3:] = R
#
# For column vectors (which is how gradients flow):
#   Y_grad = Adj^T @ grad
#
# Expanding the matrix-vector multiplication:
#   Y_grad[:3] = R^T @ (grad_t + skew(t)^T @ grad_r) = R^T @ grad_t + R^T @ cross(grad_r, t)
#   Y_grad[3:6] = R^T @ grad_r
#
# Where grad_t = grad[:3] and grad_r = grad[3:6]
# =============================================================================


# =============================================================================
# Type-specific constructor mappings
# =============================================================================

_DTYPE_TO_VEC3_CTOR = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}

_DTYPE_TO_QUAT_CTOR = {
    wp.float16: wp.quath,
    wp.float32: wp.quatf,
    wp.float64: wp.quatd,
}

_DTYPE_TO_TRANSFORM_CTOR = {
    wp.float16: wp.transformh,
    wp.float32: wp.transformf,
    wp.float64: wp.transformd,
}


def _make_mul_grad_funcs(dtype):
    """
    Factory function to create dtype-specific gradient computation functions.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        Tuple of (compute_grad_X, compute_grad_Y) warp functions
    """
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    quat_ctor = _DTYPE_TO_QUAT_CTOR[dtype]
    transform_ctor = _DTYPE_TO_TRANSFORM_CTOR[dtype]
    
    @wp.func
    def compute_grad_X(grad: T.Any) -> T.Any:
        """
        Compute gradient w.r.t. X.
        
        X_grad[:6] = grad[:6]
        X_grad[6] = 0
        """
        grad_t = wp.transform_get_translation(grad)
        grad_r = vec3_ctor(grad[3], grad[4], grad[5])
        return transform_ctor(grad_t, quat_ctor(grad_r[0], grad_r[1], grad_r[2], dtype(0.0)))

    @wp.func
    def compute_grad_Y(X: T.Any, grad: T.Any) -> T.Any:
        """
        Compute gradient w.r.t. Y.
        
        Y_grad = Adj(X)^T @ grad (in column form)
        
        For SE3_Adj(X):
            Adj = [R, skew(t)@R]
                  [0,    R    ]
        
        Adj^T = [R^T,    0   ]
                [R^T@skew(t)^T, R^T]
        
        Since skew(t)^T = -skew(t), and skew(t) @ v = t x v:
            skew(t)^T @ v = -t x v = v x t
        
        So:
            Y_grad[:3] = R^T @ grad_t
            Y_grad[3:6] = R^T @ (skew(t)^T @ grad_t + grad_r)
                        = R^T @ (cross(grad_t, t) + grad_r)
        """
        # Extract components from X
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        RT = wp.transpose(R)
        
        # Extract gradient components
        grad_t = wp.transform_get_translation(grad)
        grad_r = vec3_ctor(grad[3], grad[4], grad[5])
        
        # Compute Y_grad[:3] = R^T @ grad_t
        Y_grad_t = RT @ grad_t
        
        # Compute Y_grad[3:6] = R^T @ (cross(grad_t, t) + grad_r)
        Y_grad_r = RT @ (wp.cross(grad_t, t) + grad_r)
        
        return transform_ctor(Y_grad_t, quat_ctor(Y_grad_r[0], Y_grad_r[1], Y_grad_r[2], dtype(0.0)))

    return compute_grad_X, compute_grad_Y


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_Mul_bwd_kernel_1d(dtype):
    compute_grad_X, compute_grad_Y = _make_mul_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_Y: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad_X(grad_output[i])
        grad_Y[i] = compute_grad_Y(X[i], grad_output[i])
    return implement


def SE3_Mul_bwd_kernel_2d(dtype):
    compute_grad_X, compute_grad_Y = _make_mul_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_Y: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad_X(grad_output[i, j])
        grad_Y[i, j] = compute_grad_Y(X[i, j], grad_output[i, j])
    return implement


def SE3_Mul_bwd_kernel_3d(dtype):
    compute_grad_X, compute_grad_Y = _make_mul_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_Y: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad_X(grad_output[i, j, k])
        grad_Y[i, j, k] = compute_grad_Y(X[i, j, k], grad_output[i, j, k])
    return implement


def SE3_Mul_bwd_kernel_4d(dtype):
    compute_grad_X, compute_grad_Y = _make_mul_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_Y: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad_X(grad_output[i, j, k, l])
        grad_Y[i, j, k, l] = compute_grad_Y(X[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Mul_bwd_kernel_factories = {
    1: SE3_Mul_bwd_kernel_1d,
    2: SE3_Mul_bwd_kernel_2d,
    3: SE3_Mul_bwd_kernel_3d,
    4: SE3_Mul_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Mul_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# Import common utilities
from ...common.kernel_utils import TORCH_TO_WP_SCALAR


# =============================================================================
# Main backward function
# =============================================================================

def SE3_Mul_bwd(
    X: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_Mul.
    
    Args:
        X: First SE3 input from forward pass, shape (..., 7)
        grad_output: Gradient w.r.t. output, shape (..., 7)
        
    Returns:
        (grad_X, grad_Y): Gradients w.r.t. X and Y, each shape (..., 7)
    """
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
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
    transform_type = wp_transform_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous for warp conversion
    X = X.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=transform_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_Y_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    grad_Y_wp = wp.from_torch(grad_Y_tensor, dtype=transform_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
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

