# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.warp_functions import so3_Jl


# =============================================================================
# Backward kernel for so3_Exp
#
# The backward pass uses the left Jacobian Jl:
#   grad_input = grad_output[..., :-1] @ Jl(input)
#
# Note: grad_output has shape (..., 4) for quaternion, but only the first 3
# components (imaginary part) are used. The w component gradient is ignored
# because the constraint ||q|| = 1 means dq_w is determined by dq_xyz.
# =============================================================================


def compute_exp_grad(dtype):
    so3_Jl_impl = so3_Jl(dtype)
    
    @wp.func
    def implement(x: T.Any, grad_quat: T.Any) -> T.Any:
        """Compute grad_x for so3_Exp backward: grad_quat[:-1] @ Jl."""
        Jl = so3_Jl_impl(x)
        # Extract imaginary part of quaternion gradient (first 3 components)
        grad_xyz = wp.vector(grad_quat[0], grad_quat[1], grad_quat[2], dtype=dtype)
        # grad_input = grad_xyz @ Jl = Jl^T @ grad_xyz
        return wp.transpose(Jl) @ grad_xyz
    return implement


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def so3_Exp_bwd_kernel_1d(dtype):
    compute_exp_grad_impl = compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_x: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_x[i] = compute_exp_grad_impl(x[i], grad_output[i])
    return implement


def so3_Exp_bwd_kernel_2d(dtype):
    compute_exp_grad_impl = compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_x: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_x[i, j] = compute_exp_grad_impl(x[i, j], grad_output[i, j])
    return implement


def so3_Exp_bwd_kernel_3d(dtype):
    compute_exp_grad_impl = compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_x: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_x[i, j, k] = compute_exp_grad_impl(x[i, j, k], grad_output[i, j, k])
    return implement


def so3_Exp_bwd_kernel_4d(dtype):
    compute_exp_grad_impl = compute_exp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_x: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_x[i, j, k, l] = compute_exp_grad_impl(x[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_so3_Exp_bwd_kernel_factories = {
    1: so3_Exp_bwd_kernel_1d,
    2: so3_Exp_bwd_kernel_2d,
    3: so3_Exp_bwd_kernel_3d,
    4: so3_Exp_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _so3_Exp_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# Map torch dtype to warp scalar type for kernel specialization
_TORCH_TO_WP_SCALAR = {
    torch.float16: wp.float16,
    torch.float32: wp.float32,
    torch.float64: wp.float64,
}


# =============================================================================
# Main backward function
# =============================================================================

def so3_Exp_bwd(
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of so3_Exp with respect to input axis-angle vector.
    
    Args:
        x: Forward input (so3 axis-angle) of shape (..., 3)
        grad_output: Gradient w.r.t output quaternion of shape (..., 4)
        
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
    quat_type = wp_quat_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure contiguous
    x = x.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    x_wp = wp.from_torch(x, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=quat_type)
    
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

