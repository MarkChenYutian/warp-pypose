# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_vec3_type, wp_mat33_type
from ...common.warp_functions import so3_exp_wp_func


# =============================================================================
# Helper function: so3 (axis-angle) -> mat33 (rotation matrix)
#
# This composes:
#   1. so3 -> quaternion (via Exp)
#   2. quaternion -> rotation matrix (via quat_to_matrix)
#
# This is equivalent to Rodrigues' formula:
#   R = I + sin(θ)/θ * K + (1 - cos(θ))/θ² * K²
# where K = skew(ω), θ = ||ω||
# =============================================================================

def so3_mat_wp_func(dtype):
    so3_exp_impl = so3_exp_wp_func(dtype)
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """Convert so3 (axis-angle) to rotation matrix via quaternion."""
        # First convert to quaternion
        q = so3_exp_impl(x)
        # Then convert to rotation matrix
        return wp.quat_to_matrix(q)
    return implement


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def so3_Mat_fwd_kernel_1d(dtype):
    so3_mat_impl = so3_mat_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = so3_mat_impl(x[i])
    return implement


def so3_Mat_fwd_kernel_2d(dtype):
    so3_mat_impl = so3_mat_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = so3_mat_impl(x[i, j])
    return implement


def so3_Mat_fwd_kernel_3d(dtype):
    so3_mat_impl = so3_mat_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = so3_mat_impl(x[i, j, k])
    return implement


def so3_Mat_fwd_kernel_4d(dtype):
    so3_mat_impl = so3_mat_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = so3_mat_impl(x[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_so3_Mat_fwd_kernel_factories = {
    1: so3_Mat_fwd_kernel_1d,
    2: so3_Mat_fwd_kernel_2d,
    3: so3_Mat_fwd_kernel_3d,
    4: so3_Mat_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _so3_Mat_fwd_kernel_factories[ndim]
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

def so3_Mat_fwd(x: pp.LieTensor) -> torch.Tensor:
    """
    Convert so3 (axis-angle) to 3x3 rotation matrix.
    
    This is equivalent to PyPose's so3Type.matrix() method:
        X = input.Exp()
        I = eye(3)
        return X.Act(I).transpose(-1,-2)
    
    But more efficient as it computes directly in a single kernel.
    
    Args:
        x: so3 LieTensor of shape (..., 3) - axis-angle representation
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    x_tensor = x.tensor() if hasattr(x, 'tensor') else x
    
    # Get batch shape (everything except last dim)
    batch_shape = x_tensor.shape[:-1]
    
    ndim = len(batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        x_tensor = x_tensor.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec3_type = wp_vec3_type(dtype)
    mat33_type = wp_mat33_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    x_wp = wp.from_torch(x_tensor.contiguous(), dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_shape, 3, 3), dtype=dtype, device=x_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat33_type)
    
    # Get or create kernel for this dtype
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel with multi-dimensional grid
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=x_wp.device,
        inputs=[x_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor
