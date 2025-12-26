# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_vec3_type
from ...common.warp_functions import SO3_log_wp_func, so3_Jl_inv


# =============================================================================
# Backward kernel for SO3_Jinvp
#
# Forward: out = Jl_inv(Log(X)) @ p
#
# Given grad_output (gradient w.r.t. output), compute:
#   - grad_X: gradient w.r.t. quaternion X (shape ..., 4)
#   - grad_p: gradient w.r.t. tangent vector p (shape ..., 3)
#
# Derivation:
# Let so3 = Log(X), J = Jl_inv(so3), out = J @ p
#
# grad_p = J^T @ grad_out (straightforward from matrix-vector product)
#
# grad_so3[m] = <grad_out, d(J)/d(so3_m) @ p> for m = 0,1,2
#   where d(J)/d(so3_m) is the derivative of Jl_inv matrix w.r.t. so3_m
#
# grad_X: Use the Log backward formula:
#   grad_X = [Jl_inv(so3)^T @ grad_so3, 0] (quaternion with w=0)
#
# Jl_inv formula:
#   J = I - 0.5*K + coef2*(K@K)
#   where K = skew(so3), coef2 = (1 - θ*cot(θ/2)/2) / θ² for θ = ||so3||
#
# d(J)/d(so3_m) = -0.5*dK_m + (dcoef2/d(so3_m))*K² + coef2*(dK_m@K + K@dK_m)
#
# Skew matrix derivatives (constant matrices):
#   skew([a0,a1,a2]) = [[0,-a2,a1], [a2,0,-a0], [-a1,a0,0]]
#   dK/da0 = [[0,0,0], [0,0,-1], [0,1,0]]
#   dK/da1 = [[0,0,1], [0,0,0], [-1,0,0]]
#   dK/da2 = [[0,-1,0], [1,0,0], [0,0,0]]
# =============================================================================


def compute_jinvp_grad(dtype):
    """Generate dtype-specific backward computation for Jinvp."""
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    
    @wp.func
    def compute_grad_X_impl(q: T.Any, p: T.Any, grad_out: T.Any) -> T.Any:
        """Compute gradient w.r.t. X (quaternion)."""
        # Forward recomputation
        so3 = SO3_log_wp_func(q)
        K = wp.skew(so3)
        I = wp.identity(n=3, dtype=dtype)
        theta = wp.length(so3)
        
        eps = dtype(1e-6)
        coef2 = dtype(0.0)
        dcoef2_dtheta = dtype(0.0)
        
        if theta > eps:
            theta_half = dtype(0.5) * theta
            theta2 = theta * theta
            sin_half = wp.sin(theta_half)
            cos_half = wp.cos(theta_half)
            cot_half = cos_half / sin_half
            
            # coef2 = (1 - theta * cot_half / 2) / theta^2
            coef2 = (dtype(1.0) - theta * cot_half * dtype(0.5)) / theta2
            
            # Derivative of coef2 w.r.t. theta
            sin2_half = sin_half * sin_half
            df_dtheta = -cot_half * dtype(0.5) + theta / (dtype(4.0) * sin2_half)
            f = dtype(1.0) - theta * cot_half * dtype(0.5)
            dcoef2_dtheta = (df_dtheta * theta2 - f * dtype(2.0) * theta) / (theta2 * theta2)
        else:
            coef2 = dtype(1.0) / dtype(12.0)
            dcoef2_dtheta = dtype(0.0)
        
        # Jl_inv = I - 0.5*K + coef2*(K@K)
        Jl_inv = I - dtype(0.5) * K + coef2 * (K @ K)
        K2 = K @ K
        
        # Skew derivative matrices (inlined constants)
        # dK/da0 = [[0,0,0], [0,0,-1], [0,1,0]]
        dK0 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK0[1, 2] = dtype(-1.0)
        dK0[2, 1] = dtype(1.0)
        
        # dK/da1 = [[0,0,1], [0,0,0], [-1,0,0]]
        dK1 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK1[0, 2] = dtype(1.0)
        dK1[2, 0] = dtype(-1.0)
        
        # dK/da2 = [[0,-1,0], [1,0,0], [0,0,0]]
        dK2 = wp.matrix(shape=(3, 3), dtype=dtype)
        dK2[0, 1] = dtype(-1.0)
        dK2[1, 0] = dtype(1.0)
        
        # Compute grad_so3[m] = dot(grad_out, d(Jl_inv)/d(so3_m) @ p)
        # d(Jl_inv)/d(so3_m) = -0.5*dK_m + (dcoef2/d(so3_m))*K² + coef2*(dK_m@K + K@dK_m)
        
        # For so3[0]
        dcoef2_dso3_0 = dtype(0.0)
        if theta > eps:
            dcoef2_dso3_0 = dcoef2_dtheta * so3[0] / theta
        dJ0 = -dtype(0.5) * dK0 + dcoef2_dso3_0 * K2 + coef2 * (dK0 @ K + K @ dK0)
        grad_so3_0 = wp.dot(grad_out, dJ0 @ p)
        
        # For so3[1]
        dcoef2_dso3_1 = dtype(0.0)
        if theta > eps:
            dcoef2_dso3_1 = dcoef2_dtheta * so3[1] / theta
        dJ1 = -dtype(0.5) * dK1 + dcoef2_dso3_1 * K2 + coef2 * (dK1 @ K + K @ dK1)
        grad_so3_1 = wp.dot(grad_out, dJ1 @ p)
        
        # For so3[2]
        dcoef2_dso3_2 = dtype(0.0)
        if theta > eps:
            dcoef2_dso3_2 = dcoef2_dtheta * so3[2] / theta
        dJ2 = -dtype(0.5) * dK2 + dcoef2_dso3_2 * K2 + coef2 * (dK2 @ K + K @ dK2)
        grad_so3_2 = wp.dot(grad_out, dJ2 @ p)
        
        grad_so3 = wp.vector(grad_so3_0, grad_so3_1, grad_so3_2)
        
        # Backprop through Log: grad_X = [Jl_inv^T @ grad_so3, 0]
        grad_X_xyz = wp.transpose(Jl_inv) @ grad_so3
        grad_X = wp.quaternion(grad_X_xyz[0], grad_X_xyz[1], grad_X_xyz[2], dtype(0.0))
        
        return grad_X
    
    @wp.func
    def compute_grad_p_impl(q: T.Any, grad_out: T.Any) -> T.Any:
        """Compute grad_p = Jl_inv^T @ grad_out."""
        so3 = SO3_log_wp_func(q)
        Jl_inv = so3_Jl_inv_impl(so3)
        return wp.transpose(Jl_inv) @ grad_out
    
    return compute_grad_X_impl, compute_grad_p_impl


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SO3_Jinvp_bwd_kernel_1d(dtype):
    compute_grad_X, compute_grad_p = compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad_X(X[i], p[i], grad_output[i])
        grad_p[i] = compute_grad_p(X[i], grad_output[i])
    return implement


def SO3_Jinvp_bwd_kernel_2d(dtype):
    compute_grad_X, compute_grad_p = compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad_X(X[i, j], p[i, j], grad_output[i, j])
        grad_p[i, j] = compute_grad_p(X[i, j], grad_output[i, j])
    return implement


def SO3_Jinvp_bwd_kernel_3d(dtype):
    compute_grad_X, compute_grad_p = compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad_X(X[i, j, k], p[i, j, k], grad_output[i, j, k])
        grad_p[i, j, k] = compute_grad_p(X[i, j, k], grad_output[i, j, k])
    return implement


def SO3_Jinvp_bwd_kernel_4d(dtype):
    compute_grad_X, compute_grad_p = compute_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad_X(X[i, j, k, l], p[i, j, k, l], grad_output[i, j, k, l])
        grad_p[i, j, k, l] = compute_grad_p(X[i, j, k, l], grad_output[i, j, k, l])
    return implement


# =============================================================================
# Kernel factory selection
# =============================================================================

_SO3_Jinvp_bwd_kernel_factories = {
    1: SO3_Jinvp_bwd_kernel_1d,
    2: SO3_Jinvp_bwd_kernel_2d,
    3: SO3_Jinvp_bwd_kernel_3d,
    4: SO3_Jinvp_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SO3_Jinvp_bwd_kernel_factories[ndim]
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

def SO3_Jinvp_bwd(
    X: torch.Tensor,
    p: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SO3_Jinvp.
    
    Args:
        X: Quaternion tensor of shape (..., 4) - expanded to broadcast shape
        p: Tangent vector tensor of shape (..., 3) - expanded to broadcast shape
        grad_output: Gradient w.r.t. output, shape (..., 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 4)
        grad_p: Gradient w.r.t. p, shape (..., 3)
    """
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
        X = X.unsqueeze(0)
        p = p.unsqueeze(0)
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
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous for warp conversion
    X = X.detach().contiguous()
    p = p.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=quat_type)
    p_wp = wp.from_torch(p, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec3_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, p_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_p_tensor = grad_p_tensor.squeeze(0)
    
    return grad_X_tensor, grad_p_tensor
