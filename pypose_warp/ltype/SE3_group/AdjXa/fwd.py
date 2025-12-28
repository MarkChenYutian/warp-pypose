# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type, wp_vec3_type


# =============================================================================
# SE3_AdjXa Forward Pass
#
# SE3 element X has shape (..., 7): [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# se3 algebra element a has shape (..., 6): [v_x, v_y, v_z, w_x, w_y, w_z]
#   where v is linear velocity and w is angular velocity
#
# Adjoint action: out = Adj(X) @ a
#
# The SE3 adjoint matrix Adj(X) is 6x6:
#   Adj = [R,     skew(t) @ R]
#         [0,          R     ]
#
# where R is the 3x3 rotation matrix from quaternion and t is translation.
#
# Expanding the matrix-vector multiplication:
#   out[:3] = R @ a[:3] + skew(t) @ R @ a[3:6]
#           = R @ a[:3] + t x (R @ a[3:6])
#   out[3:6] = R @ a[3:6]
# =============================================================================


_DTYPE_TO_VEC3_CTOR = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}


def _make_se3_adjxa(dtype):
    """
    Factory function to create dtype-specific SE3 AdjXa function.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        se3_adjxa warp function
    """
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    
    @wp.func
    def se3_adjxa(X: T.Any, a_linear: T.Any, a_angular: T.Any) -> T.Any:
        """
        Compute Adj(X) @ a where X is SE3 and a is se3 (6-vector).
        
        Args:
            X: SE3 transform
            a_linear: First 3 components of a (linear velocity part)
            a_angular: Last 3 components of a (angular velocity part)
            
        Returns:
            6-component output as two vec3: (out_linear, out_angular)
        """
        # Extract rotation and translation from X
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        
        # R @ a[3:6] (angular part rotated)
        R_a_angular = R @ a_angular
        
        # out[3:6] = R @ a[3:6]
        out_angular = R_a_angular
        
        # out[:3] = R @ a[:3] + t x (R @ a[3:6])
        out_linear = R @ a_linear + wp.cross(t, R_a_angular)
        
        return out_linear, out_angular
    
    return se3_adjxa


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def SE3_AdjXa_fwd_kernel_1d(dtype):
    se3_adjxa = _make_se3_adjxa(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        a_linear: wp.array(dtype=T.Any, ndim=1),
        a_angular: wp.array(dtype=T.Any, ndim=1),
        out_linear: wp.array(dtype=T.Any, ndim=1),
        out_angular: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        ol, oa = se3_adjxa(X[i], a_linear[i], a_angular[i])
        out_linear[i] = ol
        out_angular[i] = oa
    return implement


def SE3_AdjXa_fwd_kernel_2d(dtype):
    se3_adjxa = _make_se3_adjxa(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        a_linear: wp.array(dtype=T.Any, ndim=2),
        a_angular: wp.array(dtype=T.Any, ndim=2),
        out_linear: wp.array(dtype=T.Any, ndim=2),
        out_angular: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        ol, oa = se3_adjxa(X[i, j], a_linear[i, j], a_angular[i, j])
        out_linear[i, j] = ol
        out_angular[i, j] = oa
    return implement


def SE3_AdjXa_fwd_kernel_3d(dtype):
    se3_adjxa = _make_se3_adjxa(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        a_linear: wp.array(dtype=T.Any, ndim=3),
        a_angular: wp.array(dtype=T.Any, ndim=3),
        out_linear: wp.array(dtype=T.Any, ndim=3),
        out_angular: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        ol, oa = se3_adjxa(X[i, j, k], a_linear[i, j, k], a_angular[i, j, k])
        out_linear[i, j, k] = ol
        out_angular[i, j, k] = oa
    return implement


def SE3_AdjXa_fwd_kernel_4d(dtype):
    se3_adjxa = _make_se3_adjxa(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        a_linear: wp.array(dtype=T.Any, ndim=4),
        a_angular: wp.array(dtype=T.Any, ndim=4),
        out_linear: wp.array(dtype=T.Any, ndim=4),
        out_angular: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        ol, oa = se3_adjxa(X[i, j, k, l], a_linear[i, j, k, l], a_angular[i, j, k, l])
        out_linear[i, j, k, l] = ol
        out_angular[i, j, k, l] = oa
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_AdjXa_fwd_kernel_factories = {
    1: SE3_AdjXa_fwd_kernel_1d,
    2: SE3_AdjXa_fwd_kernel_2d,
    3: SE3_AdjXa_fwd_kernel_3d,
    4: SE3_AdjXa_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_AdjXa_fwd_kernel_factories[ndim]
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

def SE3_AdjXa_fwd(X: pp.LieTensor, a: pp.LieTensor) -> torch.Tensor:
    """
    Compute Adjoint action: out = Adj(X) @ a
    
    Where Adj(X) is the 6x6 adjoint matrix of SE3:
        Adj = [R,     skew(t) @ R]
              [0,          R     ]
    
    Expanding:
        out[:3] = R @ a[:3] + t x (R @ a[3:6])
        out[3:6] = R @ a[3:6]
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        a: se3 LieTensor of shape (..., 6) - [v_x, v_y, v_z, w_x, w_y, w_z]
        
    Returns:
        Adjoint action result of shape (broadcast(...), 6)
    """
    X_tensor = X.tensor() if hasattr(X, 'tensor') else X
    a_tensor = a.tensor() if hasattr(a, 'tensor') else a
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]
    a_batch_shape = a_tensor.shape[:-1]
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, a_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X_tensor.shape} and {a_tensor.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        a_tensor = a_tensor.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*out_batch_shape, 7).contiguous()
    a_expanded = a_tensor.expand(*out_batch_shape, 6).contiguous()
    
    # Split a into linear and angular parts
    a_linear = a_expanded[..., :3].contiguous()
    a_angular = a_expanded[..., 3:6].contiguous()
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=transform_type)
    a_linear_wp = wp.from_torch(a_linear, dtype=vec3_type)
    a_angular_wp = wp.from_torch(a_angular, dtype=vec3_type)
    
    # Create output tensors
    out_linear = torch.empty((*out_batch_shape, 3), dtype=dtype, device=X_tensor.device)
    out_angular = torch.empty((*out_batch_shape, 3), dtype=dtype, device=X_tensor.device)
    out_linear_wp = wp.from_torch(out_linear, dtype=vec3_type)
    out_angular_wp = wp.from_torch(out_angular, dtype=vec3_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, a_linear_wp, a_angular_wp, out_linear_wp, out_angular_wp],
    )
    
    # Concatenate output parts
    out_tensor = torch.cat([out_linear, out_angular], dim=-1)
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor

