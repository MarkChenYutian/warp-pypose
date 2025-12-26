import warp as wp
import typing as T


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
# =============================================================================

def so3_Jl_inv(dtype):
    @wp.func
    def implement(x: T.Any) -> T.Any:
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
    return implement