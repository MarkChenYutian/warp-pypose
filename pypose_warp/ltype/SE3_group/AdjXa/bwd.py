# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    wp_vec3,
    wp_transform,
)


# =============================================================================
# SE3_AdjXa Backward Pass
#
# Forward: out = Adj(X) @ a
#   out[:3] = R @ a[:3] + t x (R @ a[3:6])
#   out[3:6] = R @ a[3:6]
#
# Backward (from PyPose SE3_AdjXa.backward):
#   X_grad[:6] = -grad @ se3_adj(out)
#   X_grad[6] = 0
#   a_grad = grad @ Adj(X)
#
# Where se3_adj is the 6x6 adjoint matrix of a Lie algebra element:
#   se3_adj(x) = [skew(x[3:6]), skew(x[:3])]
#                [    0       , skew(x[3:6])]
#
# Expanding X_grad:
#   X_grad[:3] = -(grad[:3] @ skew(out[3:6]) + grad[3:6] @ 0)
#              = -skew(out[3:6])^T @ grad[:3]
#              = skew(out[3:6]) @ grad[:3]  (since skew^T = -skew)
#              = out[3:6] x grad[:3] = -grad[:3] x out[3:6]
#   X_grad[3:6] = -(grad[:3] @ skew(out[:3]) + grad[3:6] @ skew(out[3:6]))
#               = out[:3] x grad[:3] + out[3:6] x grad[3:6]
#
# Expanding a_grad:
#   a_grad[:3] = grad[:3] @ R = R^T @ grad[:3]
#   a_grad[3:6] = grad[:3] @ skew(t) @ R + grad[3:6] @ R
#               = R^T @ (skew(t)^T @ grad[:3] + grad[3:6])
#               = R^T @ (-t x grad[:3] + grad[3:6])
#               = R^T @ (grad[:3] x t + grad[3:6])
# =============================================================================


def _make_adjxa_grad_funcs(dtype):
    """
    Factory function to create dtype-specific gradient computation functions.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        Tuple of (compute_grad_X, compute_grad_a) warp functions
    """
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_grad_X(
        out_linear: T.Any,
        out_angular: T.Any,
        grad_linear: T.Any,
        grad_angular: T.Any,
    ) -> T.Any:
        """
        Compute gradient w.r.t. X.
        
        X_grad[:3] = -grad[:3] x out[3:6] = out[3:6] x grad[:3]
        X_grad[3:6] = out[:3] x grad[:3] + out[3:6] x grad[3:6]
        X_grad[6] = 0
        """
        # X_grad[:3] = cross(out_angular, grad_linear)
        X_grad_t = wp.cross(out_angular, grad_linear)
        
        # X_grad[3:6] = cross(out_linear, grad_linear) + cross(out_angular, grad_angular)
        X_grad_r = wp.cross(out_linear, grad_linear) + wp.cross(out_angular, grad_angular)
        
        return transform_ctor(X_grad_t, quat_ctor(X_grad_r[0], X_grad_r[1], X_grad_r[2], dtype(0.0)))

    @wp.func
    def compute_grad_a(
        X: T.Any,
        grad_linear: T.Any,
        grad_angular: T.Any,
    ) -> T.Any:
        """
        Compute gradient w.r.t. a.
        
        a_grad[:3] = R^T @ grad[:3]
        a_grad[3:6] = R^T @ (cross(grad[:3], t) + grad[3:6])
        """
        # Extract rotation and translation from X
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        RT = wp.transpose(R)
        
        # a_grad[:3] = R^T @ grad_linear
        a_grad_linear = RT @ grad_linear
        
        # a_grad[3:6] = R^T @ (cross(grad_linear, t) + grad_angular)
        a_grad_angular = RT @ (wp.cross(grad_linear, t) + grad_angular)
        
        return a_grad_linear, a_grad_angular

    return compute_grad_X, compute_grad_a


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_AdjXa_bwd_kernel_1d(dtype):
    compute_grad_X, compute_grad_a = _make_adjxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out_linear: wp.array(dtype=T.Any, ndim=1),
        out_angular: wp.array(dtype=T.Any, ndim=1),
        grad_linear: wp.array(dtype=T.Any, ndim=1),
        grad_angular: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_a_linear: wp.array(dtype=T.Any, ndim=1),
        grad_a_angular: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad_X(out_linear[i], out_angular[i], grad_linear[i], grad_angular[i])
        al, aa = compute_grad_a(X[i], grad_linear[i], grad_angular[i])
        grad_a_linear[i] = al
        grad_a_angular[i] = aa
    return implement


def SE3_AdjXa_bwd_kernel_2d(dtype):
    compute_grad_X, compute_grad_a = _make_adjxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out_linear: wp.array(dtype=T.Any, ndim=2),
        out_angular: wp.array(dtype=T.Any, ndim=2),
        grad_linear: wp.array(dtype=T.Any, ndim=2),
        grad_angular: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_a_linear: wp.array(dtype=T.Any, ndim=2),
        grad_a_angular: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad_X(out_linear[i, j], out_angular[i, j], grad_linear[i, j], grad_angular[i, j])
        al, aa = compute_grad_a(X[i, j], grad_linear[i, j], grad_angular[i, j])
        grad_a_linear[i, j] = al
        grad_a_angular[i, j] = aa
    return implement


def SE3_AdjXa_bwd_kernel_3d(dtype):
    compute_grad_X, compute_grad_a = _make_adjxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out_linear: wp.array(dtype=T.Any, ndim=3),
        out_angular: wp.array(dtype=T.Any, ndim=3),
        grad_linear: wp.array(dtype=T.Any, ndim=3),
        grad_angular: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_a_linear: wp.array(dtype=T.Any, ndim=3),
        grad_a_angular: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad_X(out_linear[i, j, k], out_angular[i, j, k], grad_linear[i, j, k], grad_angular[i, j, k])
        al, aa = compute_grad_a(X[i, j, k], grad_linear[i, j, k], grad_angular[i, j, k])
        grad_a_linear[i, j, k] = al
        grad_a_angular[i, j, k] = aa
    return implement


def SE3_AdjXa_bwd_kernel_4d(dtype):
    compute_grad_X, compute_grad_a = _make_adjxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out_linear: wp.array(dtype=T.Any, ndim=4),
        out_angular: wp.array(dtype=T.Any, ndim=4),
        grad_linear: wp.array(dtype=T.Any, ndim=4),
        grad_angular: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_a_linear: wp.array(dtype=T.Any, ndim=4),
        grad_a_angular: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad_X(out_linear[i, j, k, l], out_angular[i, j, k, l], grad_linear[i, j, k, l], grad_angular[i, j, k, l])
        al, aa = compute_grad_a(X[i, j, k, l], grad_linear[i, j, k, l], grad_angular[i, j, k, l])
        grad_a_linear[i, j, k, l] = al
        grad_a_angular[i, j, k, l] = aa
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_AdjXa_bwd_kernel_factories = {
    1: SE3_AdjXa_bwd_kernel_1d,
    2: SE3_AdjXa_bwd_kernel_2d,
    3: SE3_AdjXa_bwd_kernel_3d,
    4: SE3_AdjXa_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_AdjXa_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def SE3_AdjXa_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_AdjXa.
    
    Args:
        X: SE3 input from forward pass, shape (..., 7)
        out: Output from forward pass, shape (..., 6)
        grad_output: Gradient w.r.t. output, shape (..., 6)
        
    Returns:
        (grad_X, grad_a): Gradients w.r.t. X (shape (..., 7)) and a (shape (..., 6))
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
    transform_type = wp_transform(dtype)
    vec3_type = wp_vec3(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Split out and grad_output into linear and angular parts
    out_linear = out[..., :3].contiguous()
    out_angular = out[..., 3:6].contiguous()
    grad_linear = grad_output[..., :3].contiguous()
    grad_angular = grad_output[..., 3:6].contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    out_linear_wp = wp.from_torch(out_linear, dtype=vec3_type)
    out_angular_wp = wp.from_torch(out_angular, dtype=vec3_type)
    grad_linear_wp = wp.from_torch(grad_linear, dtype=vec3_type)
    grad_angular_wp = wp.from_torch(grad_angular, dtype=vec3_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_a_linear = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    grad_a_angular = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    grad_a_linear_wp = wp.from_torch(grad_a_linear, dtype=vec3_type)
    grad_a_angular_wp = wp.from_torch(grad_a_angular, dtype=vec3_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_linear_wp, out_angular_wp, grad_linear_wp, grad_angular_wp, 
                grad_X_wp, grad_a_linear_wp, grad_a_angular_wp],
    )
    
    # Concatenate a gradients
    grad_a_tensor = torch.cat([grad_a_linear, grad_a_angular], dim=-1)
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_a_tensor = grad_a_tensor.squeeze(0)
    
    return grad_X_tensor, grad_a_tensor

