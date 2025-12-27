# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type


# =============================================================================
# SE3_Mul Forward Pass
#
# SE3 element X has shape (..., 7): [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# where t is translation (3D) and q is quaternion (4D)
#
# SE3 multiplication (composition):
#   t_out = t_X + R_X @ t_Y
#   q_out = q_X * q_Y
#
# Using Warp's transform_multiply: out = X @ Y
# =============================================================================


def _make_se3_mul(dtype):
    """
    Factory function to create dtype-specific SE3 multiplication function.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        se3_mul warp function
    """
    
    @wp.func
    def se3_mul(X: T.Any, Y: T.Any) -> T.Any:
        """Compose two SE3 transforms: out = X @ Y"""
        return wp.transform_multiply(X, Y)
    
    return se3_mul


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def SE3_Mul_fwd_kernel_1d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        Y: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_mul(X[i], Y[i])
    return implement


def SE3_Mul_fwd_kernel_2d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        Y: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_mul(X[i, j], Y[i, j])
    return implement


def SE3_Mul_fwd_kernel_3d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        Y: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_mul(X[i, j, k], Y[i, j, k])
    return implement


def SE3_Mul_fwd_kernel_4d(dtype):
    se3_mul = _make_se3_mul(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        Y: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_mul(X[i, j, k, l], Y[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Mul_fwd_kernel_factories = {
    1: SE3_Mul_fwd_kernel_1d,
    2: SE3_Mul_fwd_kernel_2d,
    3: SE3_Mul_fwd_kernel_3d,
    4: SE3_Mul_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Mul_fwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# Map torch dtype to warp scalar type for kernel specialization
_TORCH_TO_WP_SCALAR = {
    torch.float16: wp.float16,
    torch.float32: wp.float32,
    torch.float64: wp.float64,
}


# =============================================================================
# Main forward function
# =============================================================================

def SE3_Mul_fwd(X: pp.LieTensor, Y: pp.LieTensor) -> torch.Tensor:
    """
    Compose two SE3 transformations: out = X @ Y.
    
    SE3 multiplication:
        t_out = t_X + R_X @ t_Y
        q_out = q_X * q_Y
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        Y: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        
    Returns:
        Composed SE3 tensor of shape (broadcast(...), 7)
    """
    X_tensor = X.tensor()
    Y_tensor = Y.tensor() if hasattr(Y, 'tensor') else Y
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]
    Y_batch_shape = Y_tensor.shape[:-1]
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, Y_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X.shape} and {Y.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        Y_tensor = Y_tensor.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape (creates stride-0 views, no data copy)
    X_expanded = X_tensor.expand(*out_batch_shape, 7).contiguous()
    Y_expanded = Y_tensor.expand(*out_batch_shape, 7).contiguous()
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=transform_type)
    Y_wp = wp.from_torch(Y_expanded, dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 7), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=transform_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, Y_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor

