# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_VEC4,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    wp_vec4,
    wp_transform,
)


# =============================================================================
# SE3_Act4 Backward Pass
#
# Forward: out[:3] = R @ p[:3] + t * p[3], out[3] = p[3]
# where X = (t, q) is SE3 with translation t and rotation quaternion q
#
# Backward (from PyPose):
#   X_grad = grad_output @ SE3_Act4_Jacobian(out)
#   p_grad = grad_output @ SE3_Matrix4x4(X)
#
# SE3_Act4_Jacobian(out) is 4x6:
#   J[:3, :3] = I * out[3]  (translation part scaled by homogeneous coord)
#   J[:3, 3:] = skew(-out[:3])  (rotation part)
#   J[3, :] = 0
#
# SE3_Matrix4x4(X) is 4x4:
#   [[R, t], [0, 1]]
#
# In column form (Warp):
#   X_grad[:3] = grad[:3] * out[3]  (translation gradient)
#   X_grad[3:6] = cross(out[:3], grad[:3])  (rotation gradient)
#   X_grad[6] = 0
#   p_grad[:3] = R^T @ grad[:3]
#   p_grad[3] = dot(t, grad[:3]) + grad[3]
# =============================================================================


def _make_grad_funcs(dtype):
    """
    Factory function to create dtype-specific gradient computation functions.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        Tuple of (compute_se3_act4_grad_X, compute_se3_act4_grad_p) warp functions
    """
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    vec4_ctor = DTYPE_TO_VEC4[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    
    @wp.func
    def compute_se3_act4_grad_X(out: T.Any, g: T.Any) -> T.Any:
        """
        Compute SE3 gradient for Act4.
        
        Returns transform gradient: (t_grad, quaternion(rot_grad, 0))
        where:
            t_grad = grad[:3] * out[3]
            rot_grad = cross(out[:3], grad[:3])
        """
        # Extract 3D components
        out3 = vec3_ctor(out[0], out[1], out[2])
        g3 = vec3_ctor(g[0], g[1], g[2])
        pw = out[3]  # homogeneous coordinate (same as input p[3])
        
        # Translation gradient is scaled by homogeneous coordinate
        t_grad = g3 * pw
        
        # Rotation gradient is cross(out[:3], grad[:3])
        rot_grad = wp.cross(out3, g3)
        
        # Return as transform: (t_grad, quaternion(rot_grad, 0))
        return transform_ctor(t_grad, quat_ctor(rot_grad[0], rot_grad[1], rot_grad[2], dtype(0.0)))

    @wp.func
    def compute_se3_act4_grad_p(X: T.Any, g: T.Any) -> T.Any:
        """
        Compute point gradient for Act4.
        
        Returns vec4 gradient:
            p_grad[:3] = R^T @ grad[:3]
            p_grad[3] = dot(t, grad[:3]) + grad[3]
        """
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        
        # Extract grad components
        g3 = vec3_ctor(g[0], g[1], g[2])
        gw = g[3]
        
        # p_grad[:3] = R^T @ grad[:3]
        pg3 = wp.transpose(R) @ g3
        
        # p_grad[3] = dot(t, grad[:3]) + grad[3]
        pgw = wp.dot(t, g3) + gw
        
        return vec4_ctor(pg3[0], pg3[1], pg3[2], pgw)

    return compute_se3_act4_grad_X, compute_se3_act4_grad_p


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_Act4_bwd_kernel_1d(dtype):
    compute_grad_X, compute_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad_X(out[i], grad_output[i])
        grad_p[i] = compute_grad_p(X[i], grad_output[i])
    return implement


def SE3_Act4_bwd_kernel_2d(dtype):
    compute_grad_X, compute_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad_X(out[i, j], grad_output[i, j])
        grad_p[i, j] = compute_grad_p(X[i, j], grad_output[i, j])
    return implement


def SE3_Act4_bwd_kernel_3d(dtype):
    compute_grad_X, compute_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad_X(out[i, j, k], grad_output[i, j, k])
        grad_p[i, j, k] = compute_grad_p(X[i, j, k], grad_output[i, j, k])
    return implement


def SE3_Act4_bwd_kernel_4d(dtype):
    compute_grad_X, compute_grad_p = _make_grad_funcs(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad_X(out[i, j, k, l], grad_output[i, j, k, l])
        grad_p[i, j, k, l] = compute_grad_p(X[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Act4_bwd_kernel_factories = {
    1: SE3_Act4_bwd_kernel_1d,
    2: SE3_Act4_bwd_kernel_2d,
    3: SE3_Act4_bwd_kernel_3d,
    4: SE3_Act4_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Act4_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def SE3_Act4_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_Act4.
    
    Args:
        X: SE3 tensor of shape (..., 7) - expanded to broadcast shape
        out: Output from forward pass, shape (..., 4)
        grad_output: Gradient w.r.t. output, shape (..., 4)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 7)
        grad_p: Gradient w.r.t. p, shape (..., 4)
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
    vec4_type = wp_vec4(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous for warp conversion
    X = X.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    out_wp = wp.from_torch(out, dtype=vec4_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec4_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec4_type)
    
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

