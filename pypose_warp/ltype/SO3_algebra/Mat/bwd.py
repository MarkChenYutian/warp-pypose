# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_vec3_type, wp_mat33_type, wp_quat_type
from ...common.warp_functions import so3_exp_wp_func, so3_Jl


# =============================================================================
# Backward kernel for so3_Mat
#
# Forward: x (so3) -> q (quaternion) -> R (matrix)
#
# Backward uses chain rule:
#   grad_x = grad_q @ d_q_d_x
#   where grad_q = grad_R @ d_R_d_q
#
# The derivative d_R_d_q is computed inline using the same formulas as SO3_Mat_bwd.
# The derivative d_q_d_x uses the left Jacobian (same as so3_Exp_bwd).
# =============================================================================


def compute_so3_mat_grad(dtype):
    """Create a warp function to compute gradient of so3_mat."""
    so3_exp_impl = so3_exp_wp_func(dtype)
    so3_Jl_impl = so3_Jl(dtype)
    
    @wp.func
    def implement(x: T.Any, grad_R: T.Any) -> T.Any:
        """Compute grad_x for so3_Mat backward."""
        # Step 1: Compute quaternion from x (needed for d_R_d_q)
        q = so3_exp_impl(x)
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]
        
        # Step 2: Compute grad_q from grad_R using d_R_d_q
        # This is the same formula as SO3_Mat_bwd
        G = grad_R
        
        G01_plus_G10 = G[0, 1] + G[1, 0]
        G02_plus_G20 = G[0, 2] + G[2, 0]
        G12_plus_G21 = G[1, 2] + G[2, 1]
        
        G21_minus_G12 = G[2, 1] - G[1, 2]
        G02_minus_G20 = G[0, 2] - G[2, 0]
        G10_minus_G01 = G[1, 0] - G[0, 1]
        
        grad_qx = dtype(4.0) * qx * G[0, 0] + dtype(2.0) * qy * G01_plus_G10 + dtype(2.0) * qz * G02_plus_G20 + dtype(2.0) * qw * G21_minus_G12
        grad_qy = dtype(2.0) * qx * G01_plus_G10 + dtype(4.0) * qy * G[1, 1] + dtype(2.0) * qz * G12_plus_G21 + dtype(2.0) * qw * G02_minus_G20
        grad_qz = dtype(2.0) * qx * G02_plus_G20 + dtype(2.0) * qy * G12_plus_G21 + dtype(4.0) * qz * G[2, 2] + dtype(2.0) * qw * G10_minus_G01
        # grad_qw is computed but only used through Jacobian (constraint: unit quaternion)
        
        # Step 3: Compute grad_x from grad_q using d_q_d_x (left Jacobian)
        # grad_x = Jl^T @ grad_q_xyz
        Jl = so3_Jl_impl(x)
        grad_q_xyz = wp.vector(grad_qx, grad_qy, grad_qz, dtype=dtype)
        
        return wp.transpose(Jl) @ grad_q_xyz
    
    return implement


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def so3_Mat_bwd_kernel_1d(dtype):
    compute_grad_impl = compute_so3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_x: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_x[i] = compute_grad_impl(x[i], grad_output[i])
    return implement


def so3_Mat_bwd_kernel_2d(dtype):
    compute_grad_impl = compute_so3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_x: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_x[i, j] = compute_grad_impl(x[i, j], grad_output[i, j])
    return implement


def so3_Mat_bwd_kernel_3d(dtype):
    compute_grad_impl = compute_so3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_x: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_x[i, j, k] = compute_grad_impl(x[i, j, k], grad_output[i, j, k])
    return implement


def so3_Mat_bwd_kernel_4d(dtype):
    compute_grad_impl = compute_so3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_x: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_x[i, j, k, l] = compute_grad_impl(x[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_so3_Mat_bwd_kernel_factories = {
    1: so3_Mat_bwd_kernel_1d,
    2: so3_Mat_bwd_kernel_2d,
    3: so3_Mat_bwd_kernel_3d,
    4: so3_Mat_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _so3_Mat_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# Import common utilities
from ...common.kernel_utils import TORCH_TO_WP_SCALAR


# =============================================================================
# Main backward function
# =============================================================================

def so3_Mat_bwd(
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for so3_Mat.
    
    Computes gradient of rotation matrix with respect to axis-angle input
    using the chain rule through quaternion representation.
    
    Args:
        x: Forward input (so3 axis-angle) of shape (..., 3)
        grad_output: Gradient w.r.t output rotation matrix of shape (..., 3, 3)
        
    Returns:
        Gradient w.r.t input axis-angle of shape (..., 3)
    """
    batch_shape = x.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        x = x.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    dtype = x.dtype
    device = x.device
    
    vec3_type = wp_vec3_type(dtype)
    mat33_type = wp_mat33_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    x = x.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    x_wp = wp.from_torch(x, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=mat33_type)
    
    grad_x_tensor = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    grad_x_wp = wp.from_torch(grad_x_tensor, dtype=vec3_type)
    
    kernel = _get_kernel(ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=x_wp.device,
        inputs=[x_wp, grad_output_wp, grad_x_wp],
    )
    
    if squeeze_output:
        grad_x_tensor = grad_x_tensor.squeeze(0)
    
    return grad_x_tensor

