# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    wp_vec3,
    wp_transform,
)


# =============================================================================
# SE3_AdjTXa Backward Pass
#
# Forward: out = Adj(X^{-1}) @ a  (PyPose's "AdjT")
#
# Backward (from PyPose SE3_AdjTXa.backward):
#   a_grad = Adj(X) @ grad
#   X_grad[:6] = -(a @ se3_adj(a_grad))
#   X_grad[6] = 0
#
# Where se3_adj is the 6x6 adjoint matrix of a Lie algebra element:
#   se3_adj(x) = [skew(x[3:6]),  skew(x[:3])]
#                [    0       ,  skew(x[3:6])]
#
# Expanding a_grad = Adj(X) @ grad:
#   a_grad[:3] = R @ grad[:3] + t × (R @ grad[3:6])
#   a_grad[3:6] = R @ grad[3:6]
#
# Expanding X_grad = -a @ se3_adj(a_grad):
#   X_grad[:3] = -(a[:3] @ skew(a_grad[3:6]))
#              = -skew(a_grad[3:6])^T @ a[:3]
#              = skew(a_grad[3:6]) @ a[:3]  (since skew^T = -skew)
#              = a_grad[3:6] × a[:3]
#              = -a[:3] × a_grad[3:6]
#   X_grad[3:6] = -(a[:3] @ skew(a_grad[:3]) + a[3:6] @ skew(a_grad[3:6]))
#               = a_grad[:3] × a[:3] + a_grad[3:6] × a[3:6]
# =============================================================================


def _make_adjtxa_grad_funcs(dtype):
    """
    Factory function to create dtype-specific gradient computation functions.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        Tuple of (compute_grad_X, compute_grad_a) warp functions
    """
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_adjxa(X: T.Any, grad_linear: T.Any, grad_angular: T.Any) -> T.Any:
        """
        Compute a_grad = Adj(X) @ grad
        
        a_grad[:3] = R @ grad[:3] + t × (R @ grad[3:6])
        a_grad[3:6] = R @ grad[3:6]
        """
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        
        R_grad_angular = R @ grad_angular
        a_grad_angular = R_grad_angular
        a_grad_linear = R @ grad_linear + wp.cross(t, R_grad_angular)
        
        return a_grad_linear, a_grad_angular
    
    @wp.func
    def compute_grad_X(
        a_linear: T.Any,
        a_angular: T.Any,
        a_grad_linear: T.Any,
        a_grad_angular: T.Any,
    ) -> T.Any:
        """
        Compute gradient w.r.t. X.
        
        X_grad[:3] = -a[:3] × a_grad[3:6]
        X_grad[3:6] = a_grad[:3] × a[:3] + a_grad[3:6] × a[3:6]
        X_grad[6] = 0
        """
        # X_grad[:3] = -cross(a_linear, a_grad_angular)
        X_grad_t = -wp.cross(a_linear, a_grad_angular)
        
        # X_grad[3:6] = cross(a_grad_linear, a_linear) + cross(a_grad_angular, a_angular)
        X_grad_r = wp.cross(a_grad_linear, a_linear) + wp.cross(a_grad_angular, a_angular)
        
        return transform_ctor(X_grad_t, quat_ctor(X_grad_r[0], X_grad_r[1], X_grad_r[2], dtype(0.0)))

    return compute_adjxa, compute_grad_X


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_AdjTXa_bwd_kernel_1d(dtype):
    compute_adjxa, compute_grad_X = _make_adjtxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        a_linear: wp.array(dtype=T.Any, ndim=1),
        a_angular: wp.array(dtype=T.Any, ndim=1),
        grad_linear: wp.array(dtype=T.Any, ndim=1),
        grad_angular: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_a_linear: wp.array(dtype=T.Any, ndim=1),
        grad_a_angular: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        # Compute a_grad = Adj(X) @ grad
        a_grad_l, a_grad_a = compute_adjxa(X[i], grad_linear[i], grad_angular[i])
        grad_a_linear[i] = a_grad_l
        grad_a_angular[i] = a_grad_a
        # Compute X_grad = -a @ se3_adj(a_grad)
        grad_X[i] = compute_grad_X(a_linear[i], a_angular[i], a_grad_l, a_grad_a)
    return implement


def SE3_AdjTXa_bwd_kernel_2d(dtype):
    compute_adjxa, compute_grad_X = _make_adjtxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        a_linear: wp.array(dtype=T.Any, ndim=2),
        a_angular: wp.array(dtype=T.Any, ndim=2),
        grad_linear: wp.array(dtype=T.Any, ndim=2),
        grad_angular: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_a_linear: wp.array(dtype=T.Any, ndim=2),
        grad_a_angular: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        a_grad_l, a_grad_a = compute_adjxa(X[i, j], grad_linear[i, j], grad_angular[i, j])
        grad_a_linear[i, j] = a_grad_l
        grad_a_angular[i, j] = a_grad_a
        grad_X[i, j] = compute_grad_X(a_linear[i, j], a_angular[i, j], a_grad_l, a_grad_a)
    return implement


def SE3_AdjTXa_bwd_kernel_3d(dtype):
    compute_adjxa, compute_grad_X = _make_adjtxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        a_linear: wp.array(dtype=T.Any, ndim=3),
        a_angular: wp.array(dtype=T.Any, ndim=3),
        grad_linear: wp.array(dtype=T.Any, ndim=3),
        grad_angular: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_a_linear: wp.array(dtype=T.Any, ndim=3),
        grad_a_angular: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        a_grad_l, a_grad_a = compute_adjxa(X[i, j, k], grad_linear[i, j, k], grad_angular[i, j, k])
        grad_a_linear[i, j, k] = a_grad_l
        grad_a_angular[i, j, k] = a_grad_a
        grad_X[i, j, k] = compute_grad_X(a_linear[i, j, k], a_angular[i, j, k], a_grad_l, a_grad_a)
    return implement


def SE3_AdjTXa_bwd_kernel_4d(dtype):
    compute_adjxa, compute_grad_X = _make_adjtxa_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        a_linear: wp.array(dtype=T.Any, ndim=4),
        a_angular: wp.array(dtype=T.Any, ndim=4),
        grad_linear: wp.array(dtype=T.Any, ndim=4),
        grad_angular: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_a_linear: wp.array(dtype=T.Any, ndim=4),
        grad_a_angular: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        a_grad_l, a_grad_a = compute_adjxa(X[i, j, k, l], grad_linear[i, j, k, l], grad_angular[i, j, k, l])
        grad_a_linear[i, j, k, l] = a_grad_l
        grad_a_angular[i, j, k, l] = a_grad_a
        grad_X[i, j, k, l] = compute_grad_X(a_linear[i, j, k, l], a_angular[i, j, k, l], a_grad_l, a_grad_a)
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_AdjTXa_bwd_kernel_factories = {
    1: SE3_AdjTXa_bwd_kernel_1d,
    2: SE3_AdjTXa_bwd_kernel_2d,
    3: SE3_AdjTXa_bwd_kernel_3d,
    4: SE3_AdjTXa_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_AdjTXa_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def SE3_AdjTXa_bwd(
    X: torch.Tensor,
    a: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_AdjTXa.
    
    Args:
        X: SE3 input from forward pass, shape (..., 7)
        a: se3 algebra input from forward pass, shape (..., 6)
        grad_output: Gradient w.r.t. output, shape (..., 6)
        
    Returns:
        (grad_X, grad_a): Gradients w.r.t. X (shape (..., 7)) and a (shape (..., 6))
    """
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
        X = X.unsqueeze(0)
        a = a.unsqueeze(0)
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
    a = a.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Split a and grad_output into linear and angular parts
    a_linear = a[..., :3].contiguous()
    a_angular = a[..., 3:6].contiguous()
    grad_linear = grad_output[..., :3].contiguous()
    grad_angular = grad_output[..., 3:6].contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    a_linear_wp = wp.from_torch(a_linear, dtype=vec3_type)
    a_angular_wp = wp.from_torch(a_angular, dtype=vec3_type)
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
        inputs=[X_wp, a_linear_wp, a_angular_wp, grad_linear_wp, grad_angular_wp, 
                grad_X_wp, grad_a_linear_wp, grad_a_angular_wp],
    )
    
    # Concatenate a gradients
    grad_a_tensor = torch.cat([grad_a_linear, grad_a_angular], dim=-1)
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_a_tensor = grad_a_tensor.squeeze(0)
    
    return grad_X_tensor, grad_a_tensor

