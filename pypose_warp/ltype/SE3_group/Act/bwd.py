# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_vec3_type, wp_transform_type


# =============================================================================
# SE3_Act Backward Pass
#
# Forward: out = t + R @ p
# where X = (t, q) is SE3 with translation t and rotation quaternion q
#
# Backward (from PyPose):
#   X_grad[:3] = grad_output (translation gradient is identity)
#   X_grad[3:6] = cross(out, grad_output) (rotation Lie algebra gradient)
#   X_grad[6] = 0 (w component always zero)
#   p_grad = R^T @ grad_output
#
# The Jacobian of the action w.r.t. SE3 tangent space (se3) is:
#   J = [I_3x3 | skew(-out)]  (3x6 matrix)
# =============================================================================


# =============================================================================
# Type-specific constructor mappings
# =============================================================================

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


def _make_grad_funcs(dtype):
    """
    Factory function to create dtype-specific gradient computation functions.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        Tuple of (compute_se3_grad_X, compute_se3_grad_p) warp functions
    """
    # Get the correct type-specific constructors
    quat_ctor = _DTYPE_TO_QUAT_CTOR[dtype]
    transform_ctor = _DTYPE_TO_TRANSFORM_CTOR[dtype]
    
    @wp.func
    def compute_se3_grad_X(out: T.Any, g: T.Any) -> T.Any:
        """Compute SE3 gradient: (grad_output, cross(out, grad_output), 0)"""
        # Translation gradient is just the incoming gradient
        t_grad = g
        # Rotation gradient is cross(out, grad) = skew(-out) @ grad
        rot_grad = wp.cross(out, g)
        # Return as transform: (t_grad, quaternion(rot_grad, 0))
        return transform_ctor(t_grad, quat_ctor(rot_grad[0], rot_grad[1], rot_grad[2], dtype(0.0)))

    @wp.func
    def compute_se3_grad_p(X: T.Any, g: T.Any) -> T.Any:
        """Compute point gradient: R^T @ grad_output"""
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        return wp.transpose(R) @ g

    return compute_se3_grad_X, compute_se3_grad_p


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_Act_bwd_kernel_1d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_se3_grad_X(out[i], grad_output[i])
        grad_p[i] = compute_se3_grad_p(X[i], grad_output[i])
    return implement


def SE3_Act_bwd_kernel_2d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_se3_grad_X(out[i, j], grad_output[i, j])
        grad_p[i, j] = compute_se3_grad_p(X[i, j], grad_output[i, j])
    return implement


def SE3_Act_bwd_kernel_3d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_se3_grad_X(out[i, j, k], grad_output[i, j, k])
        grad_p[i, j, k] = compute_se3_grad_p(X[i, j, k], grad_output[i, j, k])
    return implement


def SE3_Act_bwd_kernel_4d(dtype):
    compute_se3_grad_X, compute_se3_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_se3_grad_X(out[i, j, k, l], grad_output[i, j, k, l])
        grad_p[i, j, k, l] = compute_se3_grad_p(X[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Act_bwd_kernel_factories = {
    1: SE3_Act_bwd_kernel_1d,
    2: SE3_Act_bwd_kernel_2d,
    3: SE3_Act_bwd_kernel_3d,
    4: SE3_Act_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Act_bwd_kernel_factories[ndim]
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

def SE3_Act_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_Act.
    
    Args:
        X: SE3 tensor of shape (..., 7) - expanded to broadcast shape
        out: Output from forward pass, shape (..., 3)
        grad_output: Gradient w.r.t. output, shape (..., 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 7)
        grad_p: Gradient w.r.t. p, shape (..., 3)
    """
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
        X = X.unsqueeze(0)
        out = out.unsqueeze(0)
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
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous for warp conversion
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    out_wp = wp.from_torch(out, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec3_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_p_tensor = grad_p_tensor.squeeze(0)
    
    return grad_X_tensor, grad_p_tensor
