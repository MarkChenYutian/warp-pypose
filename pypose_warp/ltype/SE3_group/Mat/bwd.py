# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for SE3 Mat: gradient of 4x4 transformation matrix w.r.t SE3 pose.

PyPose SE3.matrix() uses: X.unsqueeze(-2).Act(I4).transpose(-1,-2)
This means:
1. Act4 is applied to each row of 4x4 identity
2. The result is transposed

The backward uses SE3_Act4_Jacobian which computes:
- J[:3, :3] = I3x3 * p[3]  (only active for position points, not direction vectors)
- J[:3, 3:6] = skew(-p[:3])  (rotation gradient in tangent space)
- J[3, :] = 0

The total gradient sums contributions from all 4 Act4 operations.
"""

import torch
import warp as wp
import typing as T

from ...common.warp_functions import so3_Jl, so3_exp_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    DTYPE_TO_QUAT,
    DTYPE_TO_TRANSFORM,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_mat44,
    wp_transform,
)


# =============================================================================
# Helper function for computing SE3_Mat gradient using Act4 Jacobian
#
# SE3.matrix() = transpose(Act4(X, I4))
# Backward: sum of Act4 gradients for each identity row
# =============================================================================

def _make_compute_se3_mat_grad(dtype):
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    quat_ctor = DTYPE_TO_QUAT[dtype]
    transform_ctor = DTYPE_TO_TRANSFORM[dtype]
    so3_exp = so3_exp_wp_func(dtype)
    
    @wp.func
    def compute_se3_mat_grad(X: T.Any, G: T.Any) -> T.Any:
        """
        Compute SE3 gradient from 4x4 matrix gradient.
        
        Uses SE3_Act4_Jacobian formula from PyPose.
        
        Args:
            X: SE3 transform [t, q]
            G: Gradient w.r.t. 4x4 matrix
            
        Returns:
            Gradient w.r.t. SE3 pose as transform type [grad_t, grad_q]
        """
        # Extract translation and rotation matrix from X
        t = wp.transform_get_translation(X)
        q = wp.transform_get_rotation(X)
        R = wp.quat_to_matrix(q)
        
        # For SE3.matrix() = transpose(Act4(X, I4)):
        # - Before transpose: row j = Act4(X, I[j])
        # - After transpose: col j = Act4(X, I[j])
        # So grad_M[:, j] is the gradient for Act4 output j
        
        # Compute Act4 outputs for each identity column
        # out_j = [t + R @ e_j[:3], e_j[3]] where e_j is identity row j
        # For j=0,1,2: out_j = [R[:, j], 0] (direction vectors, w=0)
        # For j=3: out_j = [t, 1] (position, w=1)
        
        # SE3_Act4_Jacobian(p) where p is 4D output:
        # J[:3, :3] = I3x3 * p[3]  # translation gradient (only when w=1)
        # J[:3, 3:6] = skew(-p[:3])  # rotation gradient (tangent space)
        # J[3, :] = 0
        
        # Initialize gradient accumulators
        grad_t = vec3_ctor(dtype(0.0), dtype(0.0), dtype(0.0))
        grad_phi = vec3_ctor(dtype(0.0), dtype(0.0), dtype(0.0))
        
        # Column 0: out_0 = [R[:, 0], 0]
        # grad_out_0 = G[:, 0]
        # J_0[:3, :3] = 0 (since w=0)
        # J_0[:3, 3:6] = skew(-R[:, 0])
        # skew(-v) @ grad = grad x v = -v x grad
        out0 = vec3_ctor(R[0, 0], R[1, 0], R[2, 0])
        grad_out0 = vec3_ctor(G[0, 0], G[1, 0], G[2, 0])
        grad_phi = grad_phi + wp.cross(out0, grad_out0)  # -skew(out0)^T @ grad = out0 x grad
        
        # Column 1: out_1 = [R[:, 1], 0]
        out1 = vec3_ctor(R[0, 1], R[1, 1], R[2, 1])
        grad_out1 = vec3_ctor(G[0, 1], G[1, 1], G[2, 1])
        grad_phi = grad_phi + wp.cross(out1, grad_out1)
        
        # Column 2: out_2 = [R[:, 2], 0]
        out2 = vec3_ctor(R[0, 2], R[1, 2], R[2, 2])
        grad_out2 = vec3_ctor(G[0, 2], G[1, 2], G[2, 2])
        grad_phi = grad_phi + wp.cross(out2, grad_out2)
        
        # Column 3: out_3 = [t, 1]
        # grad_out_3 = G[:, 3]
        # J_3[:3, :3] = I3x3 (since w=1)
        # J_3[:3, 3:6] = skew(-t)
        grad_out3 = vec3_ctor(G[0, 3], G[1, 3], G[2, 3])
        grad_t = grad_out3  # I3x3 @ grad_out3 = grad_out3
        grad_phi = grad_phi + wp.cross(t, grad_out3)  # -skew(t)^T @ grad = t x grad
        
        # Convert grad_phi (tangent space) to grad_q (quaternion space)
        # Using the relationship: grad_q = 0.5 * q * grad_phi (as a pure quaternion)
        # But PyPose returns [grad_t, grad_phi, 0] and lets se3_Exp.backward handle conversion
        # So we return grad as [grad_t, grad_phi_x, grad_phi_y, grad_phi_z, 0]
        
        grad_q = quat_ctor(grad_phi[0], grad_phi[1], grad_phi[2], dtype(0.0))
        
        return transform_ctor(grad_t, grad_q)
    
    return compute_se3_mat_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad(X[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad(X[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad(X[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad(X[i, j, k, l], grad_output[i, j, k, l])
    return implement


_kernel_factories = {
    1: _make_kernel_1d,
    2: _make_kernel_2d,
    3: _make_kernel_3d,
    4: _make_kernel_4d,
}


# =============================================================================
# Main backward function
# =============================================================================

def SE3_Mat_bwd(
    X: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for SE3_Mat.
    
    Returns gradient in the form [grad_t, grad_phi, 0] where grad_phi is
    in the tangent space. This is compatible with se3_Exp backward.
    
    Args:
        X: SE3 tensor of shape (..., 7) - [tx, ty, tz, qx, qy, qz, qw]
        grad_output: Gradient w.r.t. output matrix, shape (..., 4, 4)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 7)
    """
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    transform_type = wp_transform(dtype)
    mat44_type = wp_mat44(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=transform_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=mat44_type)
    
    # Allocate output gradient
    grad_X_tensor = torch.empty((*batch_info.shape, 7), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=transform_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, grad_output_wp, grad_X_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info)
