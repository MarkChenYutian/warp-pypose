# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type, wp_mat33_type
from ...common.warp_functions import SO3_log_wp_func, so3_Jl_inv


# =============================================================================
# Higher-order function to generate dtype-specific Jinvp computation
# =============================================================================

def compute_jinvp(dtype):
    """Generate a dtype-specific Jinvp computation function."""
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    
    @wp.func
    def implement(q: T.Any, p: T.Any) -> T.Any:
        """Compute Jl_inv(Log(X)) @ p."""
        # Compute Log(X) -> so3 axis-angle
        so3 = SO3_log_wp_func(q)
        # Compute Jl_inv(so3) -> 3x3 matrix
        Jl_inv = so3_Jl_inv_impl(so3)
        # Apply: Jl_inv @ p
        return Jl_inv @ p
    return implement


# =============================================================================
# Kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SO3_Jinvp_fwd_kernel_1d(dtype):
    compute_jinvp_impl = compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        out: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        out[i] = compute_jinvp_impl(X[i], p[i])
    return implement


def SO3_Jinvp_fwd_kernel_2d(dtype):
    compute_jinvp_impl = compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        out: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        out[i, j] = compute_jinvp_impl(X[i, j], p[i, j])
    return implement


def SO3_Jinvp_fwd_kernel_3d(dtype):
    compute_jinvp_impl = compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        out: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        out[i, j, k] = compute_jinvp_impl(X[i, j, k], p[i, j, k])
    return implement


def SO3_Jinvp_fwd_kernel_4d(dtype):
    compute_jinvp_impl = compute_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        out: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        out[i, j, k, l] = compute_jinvp_impl(X[i, j, k, l], p[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_Jinvp_fwd_kernel_factories = {
    1: SO3_Jinvp_fwd_kernel_1d,
    2: SO3_Jinvp_fwd_kernel_2d,
    3: SO3_Jinvp_fwd_kernel_3d,
    4: SO3_Jinvp_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SO3_Jinvp_fwd_kernel_factories[ndim]
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

def SO3_Jinvp_fwd(X: pp.LieTensor, p: torch.Tensor) -> pp.LieTensor:
    """
    Compute Jinvp: Jl_inv(Log(X)) @ p
    
    Maps a tangent vector p through the inverse left Jacobian of the Log map.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        p: Tensor of shape (..., 3) - so3 tangent vector
        
    Returns:
        so3 LieTensor of shape (broadcast(...), 3) - transformed tangent vector
    """
    X_tensor = X.tensor()
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]  # (...,) from (..., 4)
    p_batch_shape = p.shape[:-1]         # (...,) from (..., 3)
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X.shape} and {p.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        p = p.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape (creates stride-0 views, no data copy)
    X_expanded = X_tensor.expand(*out_batch_shape, 4)
    p_expanded = p.expand(*out_batch_shape, 3)
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays (preserves strides including stride-0)
    X_wp = wp.from_torch(X_expanded, dtype=quat_type)
    p_wp = wp.from_torch(p_expanded, dtype=vec3_type)
    
    # Create output tensor and warp array
    out_tensor = torch.empty((*out_batch_shape, 3), dtype=dtype, device=X_tensor.device)
    out_wp = wp.from_torch(out_tensor, dtype=vec3_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel with multi-dimensional grid
    wp.launch(
        kernel=kernel,
        dim=out_batch_shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, out_wp],
    )
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return pp.LieTensor(out_tensor, ltype=pp.so3_type)

