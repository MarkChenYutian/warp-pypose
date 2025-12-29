# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

"""
Backward pass for se3 Mat: gradient of se3 twist to 4x4 matrix conversion.

Forward: x = [tau, phi] -> M = [R | t; 0 | 1]
where:
- t = so3_Jl(phi) @ tau
- R = quat_to_matrix(so3_exp(phi))

This is a FUSED analytical implementation that computes the gradient in a
single kernel call.

The gradient uses SE3_Act4_Jacobian formula:
- grad_t comes from the translation column of grad_M
- grad_phi comes from cross products of rotation columns with their gradients
- grad_x = se3_Jl^T @ [grad_t, grad_phi]
"""

import torch
import warp as wp
import typing as T

from ...common.warp_functions import so3_Jl, so3_exp_wp_func, se3_Jl_wp_func
from ...common.kernel_utils import (
    TORCH_TO_WP_SCALAR,
    DTYPE_TO_VEC3,
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_vec6,
    wp_mat44,
)


# =============================================================================
# Helper function for computing fused se3_Mat gradient
# =============================================================================

def _make_compute_se3_mat_grad(dtype):
    """Create fused se3_Mat backward gradient function."""
    so3_Jl_impl = so3_Jl(dtype)
    so3_exp_impl = so3_exp_wp_func(dtype)
    se3_Jl_impl = se3_Jl_wp_func(dtype)
    vec3_ctor = DTYPE_TO_VEC3[dtype]
    
    @wp.func
    def compute_se3_mat_grad(x: T.Any, G: T.Any) -> T.Any:
        """
        Compute gradient of se3_Mat w.r.t. input twist x.
        
        Args:
            x: Forward input [tau, phi] as 6D vector
            G: Gradient w.r.t. output 4x4 matrix
            
        Returns:
            Gradient w.r.t. input as 6D vector
        """
        # Extract phi (rotation part) from input
        phi = vec3_ctor(x[3], x[4], x[5])
        tau = vec3_ctor(x[0], x[1], x[2])
        
        # Compute forward quantities needed for backward
        # t = so3_Jl(phi) @ tau
        Jl_so3 = so3_Jl_impl(phi)
        t = Jl_so3 @ tau
        
        # R = quat_to_matrix(so3_exp(phi))
        q = so3_exp_impl(phi)
        R = wp.quat_to_matrix(q)
        
        # ===== Gradient computation using SE3_Act4_Jacobian =====
        # SE3.matrix() = transpose(Act4(X, I4))
        # The gradient formula uses cross products from each column.
        
        # Translation gradient (from column 3, rows 0-2)
        grad_t0 = G[0, 3]
        grad_t1 = G[1, 3]
        grad_t2 = G[2, 3]
        
        # Rotation gradient (from all 4 columns via cross products)
        # grad_phi = sum_j (out_j x grad_col_j)
        # For j=0,1,2: out_j = R[:, j]
        # For j=3: out_j = t
        
        # Column 0: out = R[:, 0]
        out0_x, out0_y, out0_z = R[0, 0], R[1, 0], R[2, 0]
        g0_x, g0_y, g0_z = G[0, 0], G[1, 0], G[2, 0]
        phi_x = out0_y * g0_z - out0_z * g0_y
        phi_y = out0_z * g0_x - out0_x * g0_z
        phi_z = out0_x * g0_y - out0_y * g0_x
        
        # Column 1: out = R[:, 1]
        out1_x, out1_y, out1_z = R[0, 1], R[1, 1], R[2, 1]
        g1_x, g1_y, g1_z = G[0, 1], G[1, 1], G[2, 1]
        phi_x = phi_x + (out1_y * g1_z - out1_z * g1_y)
        phi_y = phi_y + (out1_z * g1_x - out1_x * g1_z)
        phi_z = phi_z + (out1_x * g1_y - out1_y * g1_x)
        
        # Column 2: out = R[:, 2]
        out2_x, out2_y, out2_z = R[0, 2], R[1, 2], R[2, 2]
        g2_x, g2_y, g2_z = G[0, 2], G[1, 2], G[2, 2]
        phi_x = phi_x + (out2_y * g2_z - out2_z * g2_y)
        phi_y = phi_y + (out2_z * g2_x - out2_x * g2_z)
        phi_z = phi_z + (out2_x * g2_y - out2_y * g2_x)
        
        # Column 3: out = t
        g3_x, g3_y, g3_z = G[0, 3], G[1, 3], G[2, 3]
        phi_x = phi_x + (t[1] * g3_z - t[2] * g3_y)
        phi_y = phi_y + (t[2] * g3_x - t[0] * g3_z)
        phi_z = phi_z + (t[0] * g3_y - t[1] * g3_x)
        
        # Final step: grad_x = se3_Jl^T @ grad_SE3
        # where grad_SE3 = [grad_t, grad_phi] (6D vector)
        Jl_se3 = se3_Jl_impl(x)
        
        grad_SE3 = wp.vector(grad_t0, grad_t1, grad_t2, phi_x, phi_y, phi_z, dtype=dtype)
        grad_x = wp.transpose(Jl_se3) @ grad_SE3
        
        return grad_x
    
    return compute_se3_mat_grad


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_x: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_x[i] = compute_grad(x[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_x: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_x[i, j] = compute_grad(x[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_x: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_x[i, j, k] = compute_grad(x[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad = _make_compute_se3_mat_grad(dtype)
    
    @wp.kernel(enable_backward=False)
    def implement(
        x: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_x: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_x[i, j, k, l] = compute_grad(x[i, j, k, l], grad_output[i, j, k, l])
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

def se3_Mat_bwd(
    x: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for se3_Mat (fused analytical implementation).
    
    Computes gradient in a single kernel call using the SE3_Act4_Jacobian
    formula combined with se3_Jl.
    
    Args:
        x: Forward input (se3 twist) of shape (..., 6)
        grad_output: Gradient w.r.t output matrix of shape (..., 4, 4)
        
    Returns:
        Gradient w.r.t input twist of shape (..., 6)
    """
    x, batch_info = prepare_batch_single(x)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = x.dtype
    device = x.device
    vec6_type = wp_vec6(dtype)
    mat44_type = wp_mat44(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    x = x.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    x_wp = wp.from_torch(x, dtype=vec6_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=mat44_type)
    
    # Allocate output gradient
    grad_x_tensor = torch.empty((*batch_info.shape, 6), dtype=dtype, device=device)
    grad_x_wp = wp.from_torch(grad_x_tensor, dtype=vec6_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=x_wp.device,
        inputs=[x_wp, grad_output_wp, grad_x_wp],
    )
    
    return finalize_output(grad_x_tensor, batch_info)
