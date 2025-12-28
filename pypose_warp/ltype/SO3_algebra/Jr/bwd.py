# pyright: reportInvalidTypeForm=false
"""
Backward pass for so3 right Jacobian (Jr) computation using analytical gradients.

The backward pass computes the gradient of the loss with respect to the input x,
given the gradient of the loss with respect to the output Jr matrix.

Jr = I - coef1 * K + coef2 * K @ K

where:
    K = skew(x)
    theta = ||x||
    coef1 = (1 - cos(theta)) / theta^2
    coef2 = (theta - sin(theta)) / theta^3

The gradient is computed analytically using:
    grad_x[k] = trace(grad_output^T @ (d Jr / d x[k]))
"""

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_vec3_type, wp_mat33_type
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    prepare_batch_single,
    finalize_output,
    get_eps_for_dtype,
)


# =============================================================================
# Warp function for computing analytical gradient of Jr
# =============================================================================

def _make_so3_Jr_grad(dtype):
    """
    Create a Warp function for computing the analytical gradient of Jr.
    
    The gradient is: grad_x[k] = trace(grad_Jr^T @ (d Jr / d x[k]))
    
    d Jr / d x[k] = - (d coef1 / d x[k]) * K - coef1 * (d K / d x[k])
                    + (d coef2 / d x[k]) * K^2 + coef2 * (d K^2 / d x[k])
    
    where:
        d coef / d x[k] = (d coef / d theta) * (x[k] / theta)
        d K / d x[k] = skew(e_k) where e_k is the k-th unit vector
        d K^2 / d x[k] = (d K / d x[k]) @ K + K @ (d K / d x[k])
    """
    eps_val = get_eps_for_dtype(dtype, power=3)
    
    @wp.func
    def compute_grad(x: T.Any, grad_Jr: T.Any) -> T.Any:
        """Compute gradient of loss w.r.t. x given gradient w.r.t. Jr."""
        theta = wp.length(x)
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta2 * theta2
        
        eps = dtype(eps_val)
        
        # Compute K = skew(x) and K^2
        K = wp.skew(x)
        K2 = K @ K
        
        # Compute coefficients and their derivatives w.r.t. theta
        coef1 = dtype(0.0)
        coef2 = dtype(0.0)
        dcoef1_dtheta = dtype(0.0)
        dcoef2_dtheta = dtype(0.0)
        
        if theta > eps:
            sin_t = wp.sin(theta)
            cos_t = wp.cos(theta)
            
            # coef1 = (1 - cos(theta)) / theta^2
            coef1 = (dtype(1.0) - cos_t) / theta2
            
            # coef2 = (theta - sin(theta)) / theta^3
            coef2 = (theta - sin_t) / theta3
            
            # d coef1 / d theta = [theta*sin(theta) - 2 + 2*cos(theta)] / theta^3
            dcoef1_dtheta = (theta * sin_t - dtype(2.0) + dtype(2.0) * cos_t) / theta3
            
            # d coef2 / d theta = [3*sin(theta) - 2*theta - theta*cos(theta)] / theta^4
            dcoef2_dtheta = (dtype(3.0) * sin_t - dtype(2.0) * theta - theta * cos_t) / theta4
        else:
            # Taylor expansions for small theta
            # coef1 ≈ 0.5 - theta^2/24
            coef1 = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            
            # coef2 ≈ 1/6 - theta^2/120
            coef2 = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
            
            # d coef1 / d theta ≈ -theta/12
            dcoef1_dtheta = -theta / dtype(12.0)
            
            # d coef2 / d theta ≈ -theta/60
            dcoef2_dtheta = -theta / dtype(60.0)
        
        # Compute d theta / d x = x / theta (unit vector in direction of x)
        # For small theta, this approaches zero, so gradient contribution vanishes
        inv_theta = dtype(0.0)
        if theta > eps:
            inv_theta = dtype(1.0) / theta
        
        # Skew matrices for unit vectors (d K / d x[k] = skew(e_k))
        # dK_dx0 = skew([1,0,0]) = [[0,0,0], [0,0,-1], [0,1,0]]
        dK_dx0 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK_dx0[1, 2] = dtype(-1.0)
        dK_dx0[2, 1] = dtype(1.0)
        
        # dK_dx1 = skew([0,1,0]) = [[0,0,1], [0,0,0], [-1,0,0]]
        dK_dx1 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK_dx1[0, 2] = dtype(1.0)
        dK_dx1[2, 0] = dtype(-1.0)
        
        # dK_dx2 = skew([0,0,1]) = [[0,-1,0], [1,0,0], [0,0,0]]
        dK_dx2 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK_dx2[0, 1] = dtype(-1.0)
        dK_dx2[1, 0] = dtype(1.0)
        
        # d K^2 / d x[k] = dK_dxk @ K + K @ dK_dxk
        dK2_dx0 = dK_dx0 @ K + K @ dK_dx0
        dK2_dx1 = dK_dx1 @ K + K @ dK_dx1
        dK2_dx2 = dK_dx2 @ K + K @ dK_dx2
        
        # d coef / d x[k] = (d coef / d theta) * (x[k] / theta)
        dcoef1_dx0 = dcoef1_dtheta * x[0] * inv_theta
        dcoef1_dx1 = dcoef1_dtheta * x[1] * inv_theta
        dcoef1_dx2 = dcoef1_dtheta * x[2] * inv_theta
        
        dcoef2_dx0 = dcoef2_dtheta * x[0] * inv_theta
        dcoef2_dx1 = dcoef2_dtheta * x[1] * inv_theta
        dcoef2_dx2 = dcoef2_dtheta * x[2] * inv_theta
        
        # d Jr / d x[k] = - dcoef1_dxk * K - coef1 * dK_dxk
        #                 + dcoef2_dxk * K^2 + coef2 * dK2_dxk
        dJr_dx0 = -dcoef1_dx0 * K - coef1 * dK_dx0 + dcoef2_dx0 * K2 + coef2 * dK2_dx0
        dJr_dx1 = -dcoef1_dx1 * K - coef1 * dK_dx1 + dcoef2_dx1 * K2 + coef2 * dK2_dx1
        dJr_dx2 = -dcoef1_dx2 * K - coef1 * dK_dx2 + dcoef2_dx2 * K2 + coef2 * dK2_dx2
        
        # grad_x[k] = trace(grad_Jr^T @ dJr_dxk) = sum of element-wise product
        # Using Frobenius inner product: <A, B> = sum(A * B) = trace(A^T @ B)
        grad_x0 = dtype(0.0)
        grad_x1 = dtype(0.0)
        grad_x2 = dtype(0.0)
        
        for i in range(3):
            for j in range(3):
                grad_x0 = grad_x0 + grad_Jr[i, j] * dJr_dx0[i, j]
                grad_x1 = grad_x1 + grad_Jr[i, j] * dJr_dx1[i, j]
                grad_x2 = grad_x2 + grad_Jr[i, j] * dJr_dx2[i, j]
        
        return wp.vector(grad_x0, grad_x1, grad_x2)
    
    return compute_grad


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def _make_bwd_kernel_1d(dtype):
    compute_grad = _make_so3_Jr_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        grad_Jr: wp.array(dtype=T.Any, ndim=1),
        grad_x: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_x[i] = compute_grad(x[i], grad_Jr[i])
    return implement


def _make_bwd_kernel_2d(dtype):
    compute_grad = _make_so3_Jr_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        grad_Jr: wp.array(dtype=T.Any, ndim=2),
        grad_x: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_x[i, j] = compute_grad(x[i, j], grad_Jr[i, j])
    return implement


def _make_bwd_kernel_3d(dtype):
    compute_grad = _make_so3_Jr_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        grad_Jr: wp.array(dtype=T.Any, ndim=3),
        grad_x: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_x[i, j, k] = compute_grad(x[i, j, k], grad_Jr[i, j, k])
    return implement


def _make_bwd_kernel_4d(dtype):
    compute_grad = _make_so3_Jr_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        grad_Jr: wp.array(dtype=T.Any, ndim=4),
        grad_x: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_x[i, j, k, l] = compute_grad(x[i, j, k, l], grad_Jr[i, j, k, l])
    return implement


_bwd_kernel_factories = {
    1: _make_bwd_kernel_1d,
    2: _make_bwd_kernel_2d,
    3: _make_bwd_kernel_3d,
    4: _make_bwd_kernel_4d,
}

# Local kernel cache: (ndim, dtype) -> kernel
_bwd_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_bwd_kernel(ndim: int, dtype):
    """Get or create a backward kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _bwd_kernel_cache:
        factory = _bwd_kernel_factories[ndim]
        _bwd_kernel_cache[key] = factory(dtype)
    return _bwd_kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def so3_Jr_bwd(x_tensor: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """
    Compute the backward pass for so3 right Jacobian Jr using analytical gradients.
    
    This is more efficient than using autograd because it avoids graph construction
    and uses optimized Warp GPU kernels.
    
    Args:
        x_tensor: Input tensor of shape (..., 3)
        grad_output: Gradient of loss w.r.t. Jr output, shape (..., 3, 3)
        
    Returns:
        Gradient of loss w.r.t. x, shape (..., 3)
    """
    # Prepare batch dimensions
    x_prepared, batch_info = prepare_batch_single(x_tensor)
    
    # Also prepare grad_output - remove last two dims (3, 3) to get batch shape
    grad_batch_shape = grad_output.shape[:-2]
    if len(grad_batch_shape) == 0:
        grad_output = grad_output.unsqueeze(0)
    
    # Get warp types based on dtype
    dtype = x_tensor.dtype
    vec3_type = wp_vec3_type(dtype)
    mat33_type = wp_mat33_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    x_wp = wp.from_torch(x_prepared.contiguous(), dtype=vec3_type)
    grad_Jr_wp = wp.from_torch(grad_output.contiguous(), dtype=mat33_type)
    
    # Create output gradient tensor
    grad_x_tensor = torch.empty_like(x_prepared)
    grad_x_wp = wp.from_torch(grad_x_tensor, dtype=vec3_type)
    
    # Get kernel and launch
    kernel = _get_bwd_kernel(batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, grad_Jr_wp, grad_x_wp],
    )
    
    return finalize_output(grad_x_tensor, batch_info)
