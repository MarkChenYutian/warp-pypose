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

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    get_eps_for_dtype,
    wp_vec3,
    wp_mat33,
)


# =============================================================================
# Warp function for computing right Jacobian Jr
# =============================================================================

def _make_so3_Jr(dtype):
    """Create a Warp function for computing the right Jacobian Jr for so3."""
    # Jr divides by theta^3, so use power=3 threshold for numerical stability
    eps_val = get_eps_for_dtype(dtype, power=3)
    
    @wp.func
    def so3_Jr(x: T.Any) -> T.Any:
        """Compute right Jacobian Jr for so3."""
        theta = wp.length(x)
        K = wp.skew(x)
        I = wp.identity(n=3, dtype=dtype)
        
        eps = dtype(eps_val)
        theta2 = theta * theta
        
        coef1 = dtype(0.0)
        coef2 = dtype(0.0)
        
        if theta > eps:
            coef1 = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2 = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            coef1 = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2 = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        
        return I - coef1 * K + coef2 * (K @ K)
    return so3_Jr


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_kernel_1d(dtype):
    so3_Jr_impl = _make_so3_Jr(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = so3_Jr_impl(x[i])
    return implement


def _make_kernel_2d(dtype):
    so3_Jr_impl = _make_so3_Jr(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = so3_Jr_impl(x[i, j])
    return implement


def _make_kernel_3d(dtype):
    so3_Jr_impl = _make_so3_Jr(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = so3_Jr_impl(x[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    so3_Jr_impl = _make_so3_Jr(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = so3_Jr_impl(x[i, j, k, l])
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
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
    
    # Prepare batch dimensions
    x_tensor, batch_info = prepare_batch_single(x_tensor)
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec3_type = wp_vec3(dtype)
    mat33_type = wp_mat33(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp array
    x_wp = wp.from_torch(x_tensor.contiguous(), dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*batch_info.shape, 3, 3), dtype=dtype, device=x_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=mat33_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, out_wp],
    )
    
    return finalize_output(out_tensor, batch_info)
