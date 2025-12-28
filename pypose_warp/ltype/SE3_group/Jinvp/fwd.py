# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_transform_type, wp_vec3_type


# =============================================================================
# SE3_Jinvp Forward Pass
#
# SE3 element X has shape (..., 7): [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
# se3 algebra element p has shape (..., 6): [v_x, v_y, v_z, w_x, w_y, w_z]
#
# Jinvp computes: out = Jl_inv(Log(X)) @ p
#
# Where Log(X) maps SE3 -> se3:
#   phi = Log(quaternion) -> so3 axis-angle (3-vector)
#   tau = Jl_inv_3x3(phi) @ translation
#   log_X = [tau, phi] (6-vector)
#
# And Jl_inv_6x6 is the 6x6 inverse left Jacobian of se3:
#   Jl_inv_6x6 = [Jl_inv_3x3,  -Jl_inv_3x3 @ Q @ Jl_inv_3x3]
#                [    0     ,        Jl_inv_3x3           ]
#
# Expanding: out = Jl_inv_6x6 @ p
#   out[:3] = Jl_inv_3x3 @ p[:3] - Jl_inv_3x3 @ Q @ Jl_inv_3x3 @ p[3:6]
#   out[3:6] = Jl_inv_3x3 @ p[3:6]
# =============================================================================


_DTYPE_TO_VEC3_CTOR = {
    wp.float16: wp.vec3h,
    wp.float32: wp.vec3f,
    wp.float64: wp.vec3d,
}

_DTYPE_TO_MAT33_CTOR = {
    wp.float16: wp.mat33h,
    wp.float32: wp.mat33f,
    wp.float64: wp.mat33d,
}


def _make_se3_jinvp(dtype):
    """
    Factory function to create dtype-specific SE3_Jinvp function.
    
    Args:
        dtype: Warp scalar type (wp.float16, wp.float32, wp.float64)
        
    Returns:
        se3_jinvp warp function
    """
    vec3_ctor = _DTYPE_TO_VEC3_CTOR[dtype]
    mat33_ctor = _DTYPE_TO_MAT33_CTOR[dtype]
    
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
        
        eps = dtype(1e-6)
        coef2 = dtype(0.0)
        if theta > eps:
            theta_half = dtype(0.5) * theta
            theta2 = theta * theta
            coef2 = (dtype(1.0) - theta * wp.cos(theta_half) / (dtype(2.0) * wp.sin(theta_half))) / theta2
        else:
            coef2 = dtype(1.0) / dtype(12.0)
        
        return I - dtype(0.5) * K + coef2 * (K @ K)
    
    @wp.func
    def calcQ(tau: T.Any, phi: T.Any) -> T.Any:
        """
        Compute Q matrix used in se3_Jl_inv.
        
        Q = 0.5 * Tau + coef1 * (Phi@Tau + Tau@Phi + Phi@Tau@Phi) +
            coef2 * (Phi@Phi@Tau + Tau@Phi@Phi - 3*Phi@Tau@Phi) +
            coef3 * (Phi@Tau@Phi@Phi + Phi@Phi@Tau@Phi)
        """
        Tau = wp.skew(tau)
        Phi = wp.skew(phi)
        theta = wp.length(phi)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        
        eps = dtype(1e-6)
        
        coef1 = dtype(0.0)
        coef2 = dtype(0.0)
        coef3 = dtype(0.0)
        
        if theta > eps:
            # coef1 = (theta - sin(theta)) / theta^3
            coef1 = (theta - wp.sin(theta)) / (theta2 * theta)
            # coef2 = (theta^2 + 2*cos(theta) - 2) / (2 * theta^4)
            coef2 = (theta2 + dtype(2.0) * wp.cos(theta) - dtype(2.0)) / (dtype(2.0) * theta4)
            # coef3 = (2*theta - 3*sin(theta) + theta*cos(theta)) / (2 * theta^5)
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
    def se3_jinvp(X: T.Any, p_linear: T.Any, p_angular: T.Any) -> T.Any:
        """
        Compute Jl_inv(Log(X)) @ p where X is SE3 and p is se3 (6-vector).
        
        Args:
            X: SE3 transform
            p_linear: First 3 components of p (linear velocity part)
            p_angular: Last 3 components of p (angular velocity part)
            
        Returns:
            6-component output as two vec3: (out_linear, out_angular)
        """
        # Extract translation and rotation from X
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        
        # Step 1: Compute Log(X) -> se3 [tau, phi]
        phi = so3_log(q)
        Jl_inv_3x3 = so3_Jl_inv(phi)
        tau = Jl_inv_3x3 @ t
        
        # Step 2: Compute Q matrix
        Q = calcQ(tau, phi)
        
        # Step 3: Apply se3_Jl_inv to p
        # se3_Jl_inv = [Jl_inv_3x3, -Jl_inv_3x3 @ Q @ Jl_inv_3x3]
        #              [    0     ,        Jl_inv_3x3           ]
        
        # out[3:6] = Jl_inv_3x3 @ p[3:6]
        out_angular = Jl_inv_3x3 @ p_angular
        
        # out[:3] = Jl_inv_3x3 @ p[:3] - Jl_inv_3x3 @ Q @ Jl_inv_3x3 @ p[3:6]
        #         = Jl_inv_3x3 @ p[:3] - Jl_inv_3x3 @ Q @ out_angular
        out_linear = Jl_inv_3x3 @ p_linear - Jl_inv_3x3 @ Q @ out_angular
        
        return out_linear, out_angular
    
    return se3_jinvp


# =============================================================================
# Kernel factories for different batch dimensions (1D to 4D)
# =============================================================================

def SE3_Jinvp_fwd_kernel_1d(dtype):
    se3_jinvp = _make_se3_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        p_linear: wp.array(dtype=T.Any, ndim=1),
        p_angular: wp.array(dtype=T.Any, ndim=1),
        out_linear: wp.array(dtype=T.Any, ndim=1),
        out_angular: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        ol, oa = se3_jinvp(X[i], p_linear[i], p_angular[i])
        out_linear[i] = ol
        out_angular[i] = oa
    return implement


def SE3_Jinvp_fwd_kernel_2d(dtype):
    se3_jinvp = _make_se3_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        p_linear: wp.array(dtype=T.Any, ndim=2),
        p_angular: wp.array(dtype=T.Any, ndim=2),
        out_linear: wp.array(dtype=T.Any, ndim=2),
        out_angular: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        ol, oa = se3_jinvp(X[i, j], p_linear[i, j], p_angular[i, j])
        out_linear[i, j] = ol
        out_angular[i, j] = oa
    return implement


def SE3_Jinvp_fwd_kernel_3d(dtype):
    se3_jinvp = _make_se3_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        p_linear: wp.array(dtype=T.Any, ndim=3),
        p_angular: wp.array(dtype=T.Any, ndim=3),
        out_linear: wp.array(dtype=T.Any, ndim=3),
        out_angular: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        ol, oa = se3_jinvp(X[i, j, k], p_linear[i, j, k], p_angular[i, j, k])
        out_linear[i, j, k] = ol
        out_angular[i, j, k] = oa
    return implement


def SE3_Jinvp_fwd_kernel_4d(dtype):
    se3_jinvp = _make_se3_jinvp(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        p_linear: wp.array(dtype=T.Any, ndim=4),
        p_angular: wp.array(dtype=T.Any, ndim=4),
        out_linear: wp.array(dtype=T.Any, ndim=4),
        out_angular: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        ol, oa = se3_jinvp(X[i, j, k, l], p_linear[i, j, k, l], p_angular[i, j, k, l])
        out_linear[i, j, k, l] = ol
        out_angular[i, j, k, l] = oa
    return implement


# =============================================================================
# Kernel factory selection and caching
# =============================================================================

_SE3_Jinvp_fwd_kernel_factories = {
    1: SE3_Jinvp_fwd_kernel_1d,
    2: SE3_Jinvp_fwd_kernel_2d,
    3: SE3_Jinvp_fwd_kernel_3d,
    4: SE3_Jinvp_fwd_kernel_4d,
}

# Cache for instantiated kernels: (ndim, dtype) -> kernel
_kernel_cache: dict[tuple[int, type], T.Any] = {}


def _get_kernel(ndim: int, dtype):
    """Get or create a kernel for the given ndim and warp scalar dtype."""
    key = (ndim, dtype)
    if key not in _kernel_cache:
        factory = _SE3_Jinvp_fwd_kernel_factories[ndim]
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

def SE3_Jinvp_fwd(X: pp.LieTensor, p: pp.LieTensor) -> torch.Tensor:
    """
    Compute Jinvp: Jl_inv(Log(X)) @ p
    
    Maps a tangent vector p through the inverse left Jacobian of the Log map.
    
    Supports arbitrary batch dimensions with PyTorch-style broadcasting.
    
    Args:
        X: SE3 LieTensor of shape (..., 7) - [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        p: se3 LieTensor of shape (..., 6) - [v_x, v_y, v_z, w_x, w_y, w_z]
        
    Returns:
        Tensor of shape (broadcast(...), 6) - transformed tangent vector
    """
    X_tensor = X.tensor() if hasattr(X, 'tensor') else X
    p_tensor = p.tensor() if hasattr(p, 'tensor') else p
    
    # Get batch shapes (everything except last dim)
    X_batch_shape = X_tensor.shape[:-1]
    p_batch_shape = p_tensor.shape[:-1]
    
    # Compute broadcasted batch shape
    try:
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
    except RuntimeError as e:
        raise ValueError(
            f"Shapes {X_tensor.shape} and {p_tensor.shape} are not broadcastable: {e}"
        ) from e
    
    ndim = len(out_batch_shape)
    if ndim == 0:
        # Scalar case: add a dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        p_tensor = p_tensor.unsqueeze(0)
        out_batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {out_batch_shape}")
    
    # Expand tensors to broadcast shape
    X_expanded = X_tensor.expand(*out_batch_shape, 7).contiguous()
    p_expanded = p_tensor.expand(*out_batch_shape, 6).contiguous()
    
    # Split p into linear and angular parts
    p_linear = p_expanded[..., :3].contiguous()
    p_angular = p_expanded[..., 3:6].contiguous()
    
    # Get warp types based on dtype
    dtype = X_tensor.dtype
    transform_type = wp_transform_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    wp_scalar = _TORCH_TO_WP_SCALAR[dtype]
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_expanded, dtype=transform_type)
    p_linear_wp = wp.from_torch(p_linear, dtype=vec3_type)
    p_angular_wp = wp.from_torch(p_angular, dtype=vec3_type)
    
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
        inputs=[X_wp, p_linear_wp, p_angular_wp, out_linear_wp, out_angular_wp],
    )
    
    # Concatenate output parts
    out_tensor = torch.cat([out_linear, out_angular], dim=-1)
    
    if squeeze_output:
        out_tensor = out_tensor.squeeze(0)
    
    return out_tensor

