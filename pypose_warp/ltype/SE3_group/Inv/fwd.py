# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type


# =============================================================================
# SE3_Inv Forward Pass
#
# SE3 element X has shape (..., 7): [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# where t is translation (3D) and q is quaternion (4D)
#
# Inverse of SE3 transform:
#   q_inv = conjugate(q) = (-q_x, -q_y, -q_z, q_w)
#   t_inv = -R_inv @ t = -quat_rotate(q_inv, t)
#   Y = (t_inv, q_inv)
#
# This is equivalent to wp.transform_inverse(X)
# =============================================================================


def _make_se3_inv(dtype):
    """
    Factory function to create dtype-specific SE3 inverse function.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        se3_inv warp function
    """
    
    @wp.func
    def se3_inv(X: T.Any) -> T.Any:
        """Compute the inverse of an SE3 transform."""
        return wp.transform_inverse(X)
    
    return se3_inv


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def SE3_Inv_fwd_kernel_1d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = se3_inv(X[i])
    return implement


def SE3_Inv_fwd_kernel_2d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = se3_inv(X[i, j])
    return implement


def SE3_Inv_fwd_kernel_3d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = se3_inv(X[i, j, k])
    return implement


def SE3_Inv_fwd_kernel_4d(dtype):
    se3_inv = _make_se3_inv(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = se3_inv(X[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Inv_fwd_kernel_factories = {
    1: SE3_Inv_fwd_kernel_1d,
    2: SE3_Inv_fwd_kernel_2d,
    3: SE3_Inv_fwd_kernel_3d,
    4: SE3_Inv_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Inv_fwd_kernel_factories[ndim]
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

def SE3_Inv_fwd(X: pp.LieTensor) -> torch.Tensor:
    """
    Compute the inverse of SE3 transformation X.
    
    SE3 inverse:
        q_inv = conjugate(q)
        t_inv = -R_inv @ t
        Y = (t_inv, q_inv)
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        
    Returns:
        Inverted SE3 tensor of shape (..., 7)
    """
    X_tensor = X.tensor()
    
    # Get batch shape (everything except last dim)
    batch_shape = X_tensor.shape[:-1]
    
    ndim = len(batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Ensure contiguous for warp
    X_tensor = X_tensor.contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_tensor, dtype=transform_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=transform_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor

