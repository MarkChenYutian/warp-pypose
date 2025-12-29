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


# =============================================================================
# Helper function: calcQ for SE3 Jacobians
#
# Used in se3_Jl and se3_Jl_inv computations.
# Q = 0.5 * Tau + coef1 * (Phi@Tau + Tau@Phi + Phi@Tau@Phi) +
#     coef2 * (Phi@Phi@Tau + Tau@Phi@Phi - 3*Phi@Tau@Phi) +
#     coef3 * (Phi@Tau@Phi@Phi + Phi@Phi@Tau@Phi)
#
# where Tau = skew(tau), Phi = skew(phi), theta = ||phi||
# =============================================================================

def calcQ_wp_func(dtype):
    """Create calcQ warp function for the given dtype."""
    eps_val = get_eps_for_dtype(dtype, power=5)  # We divide by theta^5
    
    @wp.func
    def implement(tau: T.Any, phi: T.Any) -> T.Any:
        """Compute Q matrix for SE3 Jacobian computations."""
        Tau = wp.skew(tau)
        Phi = wp.skew(phi)
        theta = wp.length(phi)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        
        eps = dtype(eps_val)
        
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
            # Taylor expansions for small theta
            coef1 = dtype(1.0) / dtype(6.0) - (dtype(1.0) / dtype(120.0)) * theta2
            coef2 = dtype(1.0) / dtype(24.0) - (dtype(1.0) / dtype(720.0)) * theta2
            coef3 = dtype(1.0) / dtype(120.0) - (dtype(1.0) / dtype(2520.0)) * theta2
        
        # Q = 0.5 * Tau + coef1 * (Phi@Tau + Tau@Phi + Phi@Tau@Phi) +
        #     coef2 * (Phi@Phi@Tau + Tau@Phi@Phi - 3*Phi@Tau@Phi) +
        #     coef3 * (Phi@Tau@Phi@Phi + Phi@Phi@Tau@Phi)
        PhiTau = Phi @ Tau
        TauPhi = Tau @ Phi
        PhiPhi = Phi @ Phi
        PhiTauPhi = Phi @ TauPhi
        
        Q = dtype(0.5) * Tau
        Q = Q + coef1 * (PhiTau + TauPhi + PhiTauPhi)
        Q = Q + coef2 * (PhiPhi @ Tau + Tau @ PhiPhi - dtype(3.0) * PhiTauPhi)
        Q = Q + coef3 * (PhiTauPhi @ Phi + Phi @ PhiTauPhi)
        
        return Q
    
    return implement


# =============================================================================
# Helper function: se3_Jl_inv (6x6 left Jacobian inverse for SE3)
#
# se3_Jl_inv = [ Jl_inv           | -Jl_inv @ Q @ Jl_inv ]
#              [ 0                | Jl_inv               ]
#
# where Jl_inv is the 3x3 so3 left Jacobian inverse and Q is calcQ.
# 
# For backward, we need grad @ se3_Jl_inv which gives a 6-element vector.
# =============================================================================

def se3_Jl_inv_wp_func(dtype):
    """Create se3_Jl_inv warp function for the given dtype."""
    so3_Jl_inv_impl = so3_Jl_inv(dtype)
    calcQ_impl = calcQ_wp_func(dtype)
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """
        Compute 6x6 left Jacobian inverse for se3.
        
        Args:
            x: se3 Lie algebra element [tau(3), phi(3)] as a 6D vector
            
        Returns:
            6x6 Jacobian inverse matrix
        """
        # Extract tau and phi
        tau = wp.vector(x[0], x[1], x[2], dtype=dtype)
        phi = wp.vector(x[3], x[4], x[5], dtype=dtype)
        
        # Compute 3x3 blocks
        Jl_inv = so3_Jl_inv_impl(phi)
        Q = calcQ_impl(tau, phi)
        
        # Upper right block: -Jl_inv @ Q @ Jl_inv
        upper_right = -(Jl_inv @ Q @ Jl_inv)
        
        # Zero matrix for lower left
        Zero = wp.matrix(shape=(3, 3), dtype=dtype)
        
        # Build 6x6 matrix
        M = wp.matrix(shape=(6, 6), dtype=dtype)
        
        # Upper left: Jl_inv
        M[0, 0] = Jl_inv[0, 0]; M[0, 1] = Jl_inv[0, 1]; M[0, 2] = Jl_inv[0, 2]
        M[1, 0] = Jl_inv[1, 0]; M[1, 1] = Jl_inv[1, 1]; M[1, 2] = Jl_inv[1, 2]
        M[2, 0] = Jl_inv[2, 0]; M[2, 1] = Jl_inv[2, 1]; M[2, 2] = Jl_inv[2, 2]
        
        # Upper right: -Jl_inv @ Q @ Jl_inv
        M[0, 3] = upper_right[0, 0]; M[0, 4] = upper_right[0, 1]; M[0, 5] = upper_right[0, 2]
        M[1, 3] = upper_right[1, 0]; M[1, 4] = upper_right[1, 1]; M[1, 5] = upper_right[1, 2]
        M[2, 3] = upper_right[2, 0]; M[2, 4] = upper_right[2, 1]; M[2, 5] = upper_right[2, 2]
        
        # Lower left: Zero
        M[3, 0] = dtype(0.0); M[3, 1] = dtype(0.0); M[3, 2] = dtype(0.0)
        M[4, 0] = dtype(0.0); M[4, 1] = dtype(0.0); M[4, 2] = dtype(0.0)
        M[5, 0] = dtype(0.0); M[5, 1] = dtype(0.0); M[5, 2] = dtype(0.0)
        
        # Lower right: Jl_inv
        M[3, 3] = Jl_inv[0, 0]; M[3, 4] = Jl_inv[0, 1]; M[3, 5] = Jl_inv[0, 2]
        M[4, 3] = Jl_inv[1, 0]; M[4, 4] = Jl_inv[1, 1]; M[4, 5] = Jl_inv[1, 2]
        M[5, 3] = Jl_inv[2, 0]; M[5, 4] = Jl_inv[2, 1]; M[5, 5] = Jl_inv[2, 2]
        
        return M
    
    return implement


# =============================================================================
# Helper function: se3_Jl (6x6 left Jacobian for SE3)
#
# se3_Jl = [ so3_Jl(phi) | calcQ(tau, phi) ]
#          [ 0           | so3_Jl(phi)     ]
#
# where so3_Jl is the 3x3 left Jacobian and Q is calcQ.
#
# For backward of se3_Exp, we need: grad_input = grad_output[..., :-1] @ se3_Jl
# =============================================================================

def se3_Jl_wp_func(dtype):
    """Create se3_Jl warp function for the given dtype."""
    so3_Jl_impl = so3_Jl(dtype)
    calcQ_impl = calcQ_wp_func(dtype)
    
    @wp.func
    def implement(x: T.Any) -> T.Any:
        """
        Compute 6x6 left Jacobian for se3.
        
        Args:
            x: se3 Lie algebra element [tau(3), phi(3)] as a 6D vector
            
        Returns:
            6x6 Jacobian matrix
        """
        # Extract tau and phi
        tau = wp.vector(x[0], x[1], x[2], dtype=dtype)
        phi = wp.vector(x[3], x[4], x[5], dtype=dtype)
        
        # Compute 3x3 blocks
        J = so3_Jl_impl(phi)
        Q = calcQ_impl(tau, phi)
        
        # Build 6x6 matrix
        M = wp.matrix(shape=(6, 6), dtype=dtype)
        
        # Upper left: so3_Jl
        M[0, 0] = J[0, 0]; M[0, 1] = J[0, 1]; M[0, 2] = J[0, 2]
        M[1, 0] = J[1, 0]; M[1, 1] = J[1, 1]; M[1, 2] = J[1, 2]
        M[2, 0] = J[2, 0]; M[2, 1] = J[2, 1]; M[2, 2] = J[2, 2]
        
        # Upper right: calcQ
        M[0, 3] = Q[0, 0]; M[0, 4] = Q[0, 1]; M[0, 5] = Q[0, 2]
        M[1, 3] = Q[1, 0]; M[1, 4] = Q[1, 1]; M[1, 5] = Q[1, 2]
        M[2, 3] = Q[2, 0]; M[2, 4] = Q[2, 1]; M[2, 5] = Q[2, 2]
        
        # Lower left: Zero
        M[3, 0] = dtype(0.0); M[3, 1] = dtype(0.0); M[3, 2] = dtype(0.0)
        M[4, 0] = dtype(0.0); M[4, 1] = dtype(0.0); M[4, 2] = dtype(0.0)
        M[5, 0] = dtype(0.0); M[5, 1] = dtype(0.0); M[5, 2] = dtype(0.0)
        
        # Lower right: so3_Jl
        M[3, 3] = J[0, 0]; M[3, 4] = J[0, 1]; M[3, 5] = J[0, 2]
        M[4, 3] = J[1, 0]; M[4, 4] = J[1, 1]; M[4, 5] = J[1, 2]
        M[5, 3] = J[2, 0]; M[5, 4] = J[2, 1]; M[5, 5] = J[2, 2]
        
        return M
    
    return implement
