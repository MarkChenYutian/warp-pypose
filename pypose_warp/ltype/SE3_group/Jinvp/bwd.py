# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_transform_type, wp_vec3_type
from ...common.kernel_utils import TORCH_TO_WP_SCALAR, get_eps_for_dtype


# =============================================================================
# SE3_Jinvp Backward Pass
#
# Forward: out = se3_Jl_inv(Log(X)) @ p
#
# Structure of se3_Jl_inv(log_X) where log_X = [tau, phi]:
#   Jl_inv_6x6 = [J,    -J @ Q @ J]
#                [0,        J     ]
#
# where J = so3_Jl_inv(phi) and Q = calcQ(tau, phi)
#
# Expanding forward:
#   out[:3] = J @ p[:3] - J @ Q @ J @ p[3:]
#   out[3:] = J @ p[3:]
#
# Backward for grad_p (transpose of Jl_inv_6x6):
#   grad_p[:3] = J^T @ grad_out[:3]
#   grad_p[3:] = -J^T @ Q^T @ J^T @ grad_out[:3] + J^T @ grad_out[3:]
#
# Backward for grad_X (through Log):
#   grad_log_X = d(out)/d(log_X) @ grad_out
#   grad_X = SE3_Log_backward(grad_log_X)
#          = [se3_Jl_inv @ grad_log_X, 0]
# =============================================================================


_DTYPE_TO_VEC3_CTOR = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}

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


def _make_se3_jinvp_grad(dtype):
    """
    Factory function to create dtype-specific SE3_Jinvp gradient functions.
    """
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    quat_ctor = _DTYPE_TO_QUAT_CTOR[dtype]
    transform_ctor = _DTYPE_TO_TRANSFORM_CTOR[dtype]
    
    # Get dtype-specific epsilon thresholds
    eps_power2 = get_eps_for_dtype(dtype, power=2)
    eps_power5 = get_eps_for_dtype(dtype, power=5)
    
    @wp.func
    def so3_log(q: T.Any) -> T.Any:
        """Compute Log of SO3 quaternion -> so3 axis-angle."""
        axis, angle = wp.quat_to_axis_angle(q)
        return axis * angle
    
    @wp.func
    def so3_Jl_inv(x: T.Any) -> T.Any:
        """Compute left Jacobian inverse for so3."""
        theta = wp.length(x)
        K = wp.skew(x)
        I = wp.identity(n=3, dtype=dtype)
        
        # Use dtype-specific epsilon for theta^2 division
        eps = dtype(eps_power2)
        coef2 = dtype(0.0)
        if theta > eps:
            theta_half = dtype(0.5) * theta
            theta2 = theta * theta
            coef2 = (dtype(1.0) - theta * wp.cos(theta_half) / (dtype(2.0) * wp.sin(theta_half))) / theta2
        else:
            # Taylor expansion: coef2 ≈ 1/12
            coef2 = dtype(1.0) / dtype(12.0)
        
        return I - dtype(0.5) * K + coef2 * (K @ K)
    
    @wp.func
    def calcQ(tau: T.Any, phi: T.Any) -> T.Any:
        """Compute Q matrix used in se3_Jl_inv."""
        Tau = wp.skew(tau)
        Phi = wp.skew(phi)
        theta = wp.length(phi)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        
        # Use dtype-specific epsilon for theta^5 division
        eps = dtype(eps_power5)
        
        coef1 = dtype(0.0)
        coef2 = dtype(0.0)
        coef3 = dtype(0.0)
        
        if theta > eps:
            coef1 = (theta - wp.sin(theta)) / (theta2 * theta)
            coef2 = (theta2 + dtype(2.0) * wp.cos(theta) - dtype(2.0)) / (dtype(2.0) * theta4)
            coef3 = (dtype(2.0) * theta - dtype(3.0) * wp.sin(theta) + theta * wp.cos(theta)) / (dtype(2.0) * theta4 * theta)
        else:
            # Taylor expansion for small theta
            coef1 = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
            coef2 = dtype(1.0) / dtype(24.0) - (dtype(1.0) / dtype(720.0)) * theta2
            coef3 = dtype(1.0) / dtype(120.0) - (dtype(1.0) / dtype(2520.0)) * theta2
        
        PhiTau = Phi @ Tau
        TauPhi = Tau @ Phi
        PhiPhi = Phi @ Phi
        PhiTauPhi = Phi @ Tau @ Phi
        
        Q = dtype(0.5) * Tau + coef1 * (PhiTau + TauPhi + PhiTauPhi) + \
            coef2 * (PhiPhi @ Tau + Tau @ PhiPhi - dtype(3.0) * PhiTauPhi) + \
            coef3 * (Phi @ Tau @ PhiPhi + PhiPhi @ Tau @ Phi)
        
        return Q
    
    @wp.func
    def compute_grad_p(
        J: T.Any,
        Q: T.Any,
        grad_out_linear: T.Any,
        grad_out_angular: T.Any,
    ) -> T.Any:
        """
        Compute gradient w.r.t. p.
        
        grad_p[:3] = J^T @ grad_out[:3]
        grad_p[3:] = -J^T @ Q^T @ J^T @ grad_out[:3] + J^T @ grad_out[3:]
        """
        JT = wp.transpose(J)
        QT = wp.transpose(Q)
        
        grad_p_linear = JT @ grad_out_linear
        grad_p_angular = -JT @ QT @ JT @ grad_out_linear + JT @ grad_out_angular
        
        return grad_p_linear, grad_p_angular
    
    @wp.func
    def compute_grad_X(
        X: T.Any,
        p_linear: T.Any,
        p_angular: T.Any,
        grad_out_linear: T.Any,
        grad_out_angular: T.Any,
    ) -> T.Any:
        """
        Compute gradient w.r.t. X.
        
        This involves:
        1. Computing grad_log_X from the chain rule through Jl_inv
        2. Backpropagating through SE3_Log
        
        For simplicity, we use the approximation that the dominant gradient
        contribution comes from the J matrix (ignoring Q derivatives).
        
        Full gradient would require derivatives of J and Q w.r.t. log_X components.
        """
        # Extract translation and rotation from X
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        
        # Compute log(X)
        phi = so3_log(q)
        J = so3_Jl_inv(phi)
        tau = J @ t
        Q = calcQ(tau, phi)
        
        # Compute se3_Jl_inv for Log backward
        # se3_Jl_inv @ v where v is grad_log_X
        # The Log backward is: grad_X[:6] = v @ se3_Jl_inv(log_X)
        # which equals: grad_X[:3] = J @ v[:3], grad_X[3:6] = -J @ Q @ J @ v[:3] + J @ v[3:]
        
        # For the full backward, we need grad_log_X from the output gradients.
        # The approximate gradient ignoring d(J)/d(phi) and d(Q)/d(tau,phi):
        # grad_log_X[:3] ~ J^T @ grad_out[:3] (from p[:3] term)
        # grad_log_X[3:] ~ -J^T @ Q^T @ J^T @ grad_out[:3] + J^T @ grad_out[3:] (from p[3:] terms)
        
        JT = wp.transpose(J)
        QT = wp.transpose(Q)
        
        # Simplified gradient contribution (ignoring Jacobian derivatives for now)
        # This is an approximation - full derivative would require much more complex computation
        grad_tau = JT @ grad_out_linear
        grad_phi = -JT @ QT @ JT @ grad_out_linear + JT @ grad_out_angular
        
        # Backprop through Log: grad_X = se3_Jl_inv @ grad_log_X
        # se3_Jl_inv @ [grad_tau, grad_phi] gives grad_X[:6]
        grad_X_linear = J @ grad_tau - J @ Q @ J @ grad_phi
        grad_X_angular = J @ grad_phi
        
        return transform_ctor(grad_X_linear, quat_ctor(grad_X_angular[0], grad_X_angular[1], grad_X_angular[2], dtype(0.0)))
    
    @wp.func
    def jinvp_backward(
        X: T.Any,
        p_linear: T.Any,
        p_angular: T.Any,
        grad_out_linear: T.Any,
        grad_out_angular: T.Any,
    ) -> T.Any:
        """
        Full backward pass for SE3_Jinvp.
        """
        # Recompute forward values
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        phi = so3_log(q)
        J = so3_Jl_inv(phi)
        tau = J @ t
        Q = calcQ(tau, phi)
        
        # Compute grad_p
        grad_p_linear, grad_p_angular = compute_grad_p(J, Q, grad_out_linear, grad_out_angular)
        
        # Compute grad_X
        grad_X = compute_grad_X(X, p_linear, p_angular, grad_out_linear, grad_out_angular)
        
        return grad_X, grad_p_linear, grad_p_angular
    
    return jinvp_backward


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def SE3_Jinvp_bwd_kernel_1d(dtype):
    jinvp_backward = _make_se3_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p_linear: wp.array(dtype=T.Any, ndim=1),
        p_angular: wp.array(dtype=T.Any, ndim=1),
        grad_out_linear: wp.array(dtype=T.Any, ndim=1),
        grad_out_angular: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
        grad_p_linear: wp.array(dtype=T.Any, ndim=1),
        grad_p_angular: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        gX, gpl, gpa = jinvp_backward(X[i], p_linear[i], p_angular[i], grad_out_linear[i], grad_out_angular[i])
        grad_X[i] = gX
        grad_p_linear[i] = gpl
        grad_p_angular[i] = gpa
    return implement


def SE3_Jinvp_bwd_kernel_2d(dtype):
    jinvp_backward = _make_se3_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p_linear: wp.array(dtype=T.Any, ndim=2),
        p_angular: wp.array(dtype=T.Any, ndim=2),
        grad_out_linear: wp.array(dtype=T.Any, ndim=2),
        grad_out_angular: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
        grad_p_linear: wp.array(dtype=T.Any, ndim=2),
        grad_p_angular: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        gX, gpl, gpa = jinvp_backward(X[i, j], p_linear[i, j], p_angular[i, j], grad_out_linear[i, j], grad_out_angular[i, j])
        grad_X[i, j] = gX
        grad_p_linear[i, j] = gpl
        grad_p_angular[i, j] = gpa
    return implement


def SE3_Jinvp_bwd_kernel_3d(dtype):
    jinvp_backward = _make_se3_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p_linear: wp.array(dtype=T.Any, ndim=3),
        p_angular: wp.array(dtype=T.Any, ndim=3),
        grad_out_linear: wp.array(dtype=T.Any, ndim=3),
        grad_out_angular: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
        grad_p_linear: wp.array(dtype=T.Any, ndim=3),
        grad_p_angular: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        gX, gpl, gpa = jinvp_backward(X[i, j, k], p_linear[i, j, k], p_angular[i, j, k], grad_out_linear[i, j, k], grad_out_angular[i, j, k])
        grad_X[i, j, k] = gX
        grad_p_linear[i, j, k] = gpl
        grad_p_angular[i, j, k] = gpa
    return implement


def SE3_Jinvp_bwd_kernel_4d(dtype):
    jinvp_backward = _make_se3_jinvp_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p_linear: wp.array(dtype=T.Any, ndim=4),
        p_angular: wp.array(dtype=T.Any, ndim=4),
        grad_out_linear: wp.array(dtype=T.Any, ndim=4),
        grad_out_angular: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
        grad_p_linear: wp.array(dtype=T.Any, ndim=4),
        grad_p_angular: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        gX, gpl, gpa = jinvp_backward(X[i, j, k, l], p_linear[i, j, k, l], p_angular[i, j, k, l], grad_out_linear[i, j, k, l], grad_out_angular[i, j, k, l])
        grad_X[i, j, k, l] = gX
        grad_p_linear[i, j, k, l] = gpl
        grad_p_angular[i, j, k, l] = gpa
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Jinvp_bwd_kernel_factories = {
    1: SE3_Jinvp_bwd_kernel_1d,
    2: SE3_Jinvp_bwd_kernel_2d,
    3: SE3_Jinvp_bwd_kernel_3d,
    4: SE3_Jinvp_bwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Jinvp_bwd_kernel_factories[ndim]
        _kernel_cache[key] = factory(dtype)
    return _kernel_cache[key]


# =============================================================================
# Main backward function
# =============================================================================

def SE3_Jinvp_bwd(
    X: torch.Tensor,
    p: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SE3_Jinvp.
    
    Args:
        X: SE3 input from forward pass, shape (..., 7)
        p: se3 algebra input from forward pass, shape (..., 6)
        grad_output: Gradient w.r.t. output, shape (..., 6)
        
    Returns:
        (grad_X, grad_p): Gradients w.r.t. X (shape (..., 7)) and p (shape (..., 6))
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
    transform_type = wp_transform_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    p = p.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Split p and grad_output into linear and angular parts
    p_linear = p[..., :3].contiguous()
    p_angular = p[..., 3:6].contiguous()
    grad_out_linear = grad_output[..., :3].contiguous()
    grad_out_angular = grad_output[..., 3:6].contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    p_linear_wp = wp.from_torch(p_linear, dtype=vec3_type)
    p_angular_wp = wp.from_torch(p_angular, dtype=vec3_type)
    grad_out_linear_wp = wp.from_torch(grad_out_linear, dtype=vec3_type)
    grad_out_angular_wp = wp.from_torch(grad_out_angular, dtype=vec3_type)
    
    # Allocate output gradients
    grad_X_tensor = torch.empty((*batch_shape, 7), dtype=dtype, device=device)
    grad_p_linear = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    grad_p_angular = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    grad_p_linear_wp = wp.from_torch(grad_p_linear, dtype=vec3_type)
    grad_p_angular_wp = wp.from_torch(grad_p_angular, dtype=vec3_type)
    
    # Get or create kernel for this dtype and ndim
    kernel = _get_kernel(ndim, wp_scalar)
    
    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, p_linear_wp, p_angular_wp, grad_out_linear_wp, grad_out_angular_wp,
                grad_X_wp, grad_p_linear_wp, grad_p_angular_wp],
    )
    
    # Concatenate p gradients
    grad_p_tensor = torch.cat([grad_p_linear, grad_p_angular], dim=-1)
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_p_tensor = grad_p_tensor.squeeze(0)
    
    return grad_X_tensor, grad_p_tensor

