import warp as wp
import typing as T

from .kernel_utils import get_eps_for_dtype


# =============================================================================
# Helper function: quaternion -> axis-angle (so3 Lie algebra)
# =============================================================================

@wp.func
def SO3_log_wp_func(q: T.Any):
    axis, angle = wp.quat_to_axis_angle(q)
    return axis * angle


# =============================================================================
# Helper function for computing so3_Jl_inv (left Jacobian inverse)
#
# Formula: Jl_inv = I - 0.5 * K + coef2 * (K @ K)
# where:
#   K = skew(x) - 3x3 skew-symmetric matrix
#   theta = ||x||
#   coef2 = (1 - theta * cos(theta/2) / (2 * sin(theta/2))) / theta^2  if theta > eps
#   coef2 = 1/12  otherwise (Taylor expansion)
#
# Note: Uses dtype-specific epsilon to avoid FP16 underflow in theta^2 division.
# =============================================================================

def so3_Jl_inv(dtype):
    # Get appropriate epsilon for theta^2 division
    eps_val = get_eps_for_dtype(dtype, power=2)
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """Compute left Jacobian inverse for so3."""
        theta = wp.length(x)
        K = wp.skew(x)
        I = wp.identity(n=3, dtype=dtype)
        
        eps = dtype(eps_val)
        coef2 = dtype(0.0)
        if theta > eps:
            theta_half = dtype(0.5) * theta
            theta2 = theta * theta
            coef2 = (dtype(1.0) - theta * wp.cos(theta_half) / (dtype(2.0) * wp.sin(theta_half))) / theta2
        else:
            # Taylor expansion: coef2 ≈ 1/12 + O(theta^2)
            coef2 = dtype(1.0) / dtype(12.0)
        
        return I - dtype(0.5) * K + coef2 * (K @ K)
    return implement


# =============================================================================
# Helper function for computing so3_Jl (left Jacobian)
#
# Formula: Jl = I + coef1 * K + coef2 * (K @ K)
# where:
#   K = skew(x) - 3x3 skew-symmetric matrix
#   theta = ||x||
#   coef1 = (1 - cos(theta)) / theta^2  if theta > eps
#   coef1 = 0.5 - (1/24) * theta^2  otherwise (Taylor expansion)
#   coef2 = (theta - sin(theta)) / theta^3  if theta > eps
#   coef2 = 1/6 - (1/120) * theta^2  otherwise (Taylor expansion)
#
# Note: Uses dtype-specific epsilon (power=3) to avoid FP16 underflow.
# =============================================================================

def so3_Jl(dtype):
    # Get appropriate epsilon for theta^3 division (the stricter requirement)
    eps_val = get_eps_for_dtype(dtype, power=3)
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """Compute left Jacobian for so3."""
        theta = wp.length(x)
        K = wp.skew(x)
        I = wp.identity(n=3, dtype=dtype)
        
        eps = dtype(eps_val)
        theta2 = theta * theta
        
        coef1 = dtype(0.0)
        coef2 = dtype(0.0)
        
        if theta > eps:
            coef1 = (dtype(1.0) - wp.cos(theta)) / theta2
            coef2 = (theta - wp.sin(theta)) / (theta * theta2)
        else:
            # Taylor expansion for small theta
            coef1 = dtype(0.5) - (dtype(1.0) / dtype(24.0)) * theta2
            coef2 = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
        
        return I + coef1 * K + coef2 * (K @ K)
    return implement


# =============================================================================
# Helper function: so3 (axis-angle) -> quaternion (Exp map)
#
# For axis-angle x with theta = ||x||:
#   quaternion = (x * sin(theta/2) / theta, cos(theta/2))
# For small theta, use Taylor expansion.
#
# Note: Uses dtype-specific epsilon for division by theta.
# =============================================================================

def so3_exp_wp_func(dtype):
    # Division by theta only, so power=2 threshold is sufficient
    eps_val = get_eps_for_dtype(dtype, power=2)
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """Compute exponential map from so3 to SO3 (quaternion)."""
        theta = wp.length(x)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta_half = dtype(0.5) * theta
        
        eps = dtype(eps_val)
        
        imag_factor = dtype(0.0)
        real_factor = dtype(0.0)
        
        if theta > eps:
            imag_factor = wp.sin(theta_half) / theta
            real_factor = wp.cos(theta_half)
        else:
            # Taylor expansion for small theta
            # sin(theta/2) / theta ≈ 0.5 - theta^2/48 + theta^4/3840
            # cos(theta/2) ≈ 1 - theta^2/8 + theta^4/384
            imag_factor = dtype(0.5) - (dtype(1.0) / dtype(48.0)) * theta2 + (dtype(1.0) / dtype(3840.0)) * theta4
            real_factor = dtype(1.0) - (dtype(1.0) / dtype(8.0)) * theta2 + (dtype(1.0) / dtype(384.0)) * theta4
        
        # Quaternion format: (x, y, z, w)
        qx = x[0] * imag_factor
        qy = x[1] * imag_factor
        qz = x[2] * imag_factor
        qw = real_factor
        
        return wp.quaternion(qx, qy, qz, qw)
    return implement
