# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Forward pass for so3 right Jacobian (Jr) computation.

The right Jacobian Jr of so3 is defined as:
    Jr = I - (1 - cos(theta)) / theta^2 * K + (theta - sin(theta)) / theta^3 * K @ K

where:
    K = skew(x)  - 3x3 skew-symmetric matrix
    theta = ||x||

For numerical stability, when theta < eps, Jr ≈ I (identity matrix).
"""

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_vec3_type, wp_mat33_type


# =============================================================================
# Warp function for computing right Jacobian Jr
# =============================================================================

def so3_Jr_wp_func(dtype):
    """
    Create a Warp function for computing the right Jacobian Jr for so3.
    
    Formula: Jr = I - coef1 * K + coef2 * K @ K
    where:
        coef1 = (1 - cos(theta)) / theta^2  if theta > eps
        coef1 = 0.5 - (1/24) * theta^2  otherwise (Taylor expansion)
        coef2 = (theta - sin(theta)) / theta^3  if theta > eps
        coef2 = 1/6 - (1/120) * theta^2  otherwise (Taylor expansion)
    """
    # Dtype-dependent epsilon for numerical stability
    # These are approximate machine epsilon values for each dtype
    if dtype == wp.float16:
        eps_val = 1e-3  # fp16 has very limited precision, use larger threshold
    elif dtype == wp.float32:
        eps_val = 1e-6
    else:  # fp64
        eps_val = 1e-12
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """Compute right Jacobian Jr for so3."""
        theta = wp.length(x)
        K = wp.skew(x)
        I = wp.identity(n=3, dtype=dtype)
        
        # Use dtype-appropriate epsilon for numerical stability
        eps = dtype(eps_val)
        theta2 = theta * theta
        
        coef1 = dtype(0.0)
        coef2 = dtype(0.0)
        
        if theta > eps:
            # Standard formula
            coef1 = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2 = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            # Taylor expansion for small theta (better numeric stability)
            coef1 = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2 = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        
        # Jr = I - coef1 * K + coef2 * K @ K
        # Note: negative sign on coef1 term (unlike Jl which has positive)
        return I - coef1 * K + coef2 * (K @ K)
    return implement


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def so3_Jr_fwd_kernel_1d(dtype):
    so3_Jr_impl = so3_Jr_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = so3_Jr_impl(x[i])
    return implement


def so3_Jr_fwd_kernel_2d(dtype):
    so3_Jr_impl = so3_Jr_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = so3_Jr_impl(x[i, j])
    return implement


def so3_Jr_fwd_kernel_3d(dtype):
    so3_Jr_impl = so3_Jr_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = so3_Jr_impl(x[i, j, k])
    return implement


def so3_Jr_fwd_kernel_4d(dtype):
    so3_Jr_impl = so3_Jr_wp_func(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = so3_Jr_impl(x[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_so3_Jr_fwd_kernel_factories = {
    1: so3_Jr_fwd_kernel_1d,
    2: so3_Jr_fwd_kernel_2d,
    3: so3_Jr_fwd_kernel_3d,
    4: so3_Jr_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _so3_Jr_fwd_kernel_factories[ndim]
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

def so3_Jr_fwd(x: pp.LieTensor) -> torch.Tensor:
    """
    Compute the right Jacobian Jr of so3.
    
    The right Jacobian Jr is the derivative of the exponential map:
        d/dx [Exp(x + dx)] = Exp(x) @ Jr(x) @ dx (at dx=0)
    
    Args:
        x: so3 LieTensor of shape (..., 3) - axis-angle representation
        
    Returns:
        Right Jacobian tensor of shape (..., 3, 3)
    """
    x_tensor = x.tensor()
    
    # Get batch shape (everything except last dim which is 3 for so3)
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

