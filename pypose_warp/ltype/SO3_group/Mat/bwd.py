# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ...common.kernel_utils import (
    KernelRegistry,
    prepare_batch_single,
    finalize_output,
    wp_quat,
    wp_mat33,
)


# =============================================================================
# Backward kernel for SO3_Mat
#
# Forward: R = quat_to_matrix(q) using Warp's formula.
#
# Warp's quat_to_matrix uses quat_rotate on identity matrix columns.
# The formula for quat_rotate gives:
#   R[0,0] = 2*w² + 2*x² - 1
#   R[0,1] = 2*x*y - 2*w*z
#   R[0,2] = 2*x*z + 2*w*y
#   R[1,0] = 2*x*y + 2*w*z
#   R[1,1] = 2*w² + 2*y² - 1
#   R[1,2] = 2*y*z - 2*w*x
#   R[2,0] = 2*x*z - 2*w*y
#   R[2,1] = 2*y*z + 2*w*x
#   R[2,2] = 2*w² + 2*z² - 1
#
# Given grad_R (gradient w.r.t. matrix elements), compute grad_q:
#   grad_x = 4*x*G00 + 2*y*(G01 + G10) + 2*z*(G02 + G20) + 2*w*(G21 - G12)
#   grad_y = 2*x*(G01 + G10) + 4*y*G11 + 2*z*(G12 + G21) + 2*w*(G02 - G20)
#   grad_z = 2*x*(G02 + G20) + 2*y*(G12 + G21) + 4*z*G22 + 2*w*(G10 - G01)
#   grad_w = 4*w*(G00 + G11 + G22) + 2*x*(G21 - G12) + 2*y*(G02 - G20) + 2*z*(G10 - G01)
# =============================================================================

def _make_compute_grad_q(dtype):
    @wp.func
    def compute_grad_q(q: T.Any, G: T.Any) -> T.Any:
        """Compute quaternion gradient from rotation matrix gradient."""
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        
        # Symmetric and antisymmetric combinations
        G01_plus_G10 = G[0, 1] + G[1, 0]
        G02_plus_G20 = G[0, 2] + G[2, 0]
        G12_plus_G21 = G[1, 2] + G[2, 1]
        
        G21_minus_G12 = G[2, 1] - G[1, 2]
        G02_minus_G20 = G[0, 2] - G[2, 0]
        G10_minus_G01 = G[1, 0] - G[0, 1]
        
        grad_x = dtype(4.0) * x * G[0, 0] + dtype(2.0) * y * G01_plus_G10 + dtype(2.0) * z * G02_plus_G20 + dtype(2.0) * w * G21_minus_G12
        grad_y = dtype(2.0) * x * G01_plus_G10 + dtype(4.0) * y * G[1, 1] + dtype(2.0) * z * G12_plus_G21 + dtype(2.0) * w * G02_minus_G20
        grad_z = dtype(2.0) * x * G02_plus_G20 + dtype(2.0) * y * G12_plus_G21 + dtype(4.0) * z * G[2, 2] + dtype(2.0) * w * G10_minus_G01
        grad_w = dtype(4.0) * w * (G[0, 0] + G[1, 1] + G[2, 2]) + dtype(2.0) * x * G21_minus_G12 + dtype(2.0) * y * G02_minus_G20 + dtype(2.0) * z * G10_minus_G01
        
        return wp.quaternion(grad_x, grad_y, grad_z, grad_w)
    return compute_grad_q


# =============================================================================
# Backward kernel factories for 1D to 4D batch dimensions
# =============================================================================

def _make_kernel_1d(dtype):
    compute_grad_q = _make_compute_grad_q(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=1),
        grad_output: wp.array(dtype=T.Any, ndim=1),
        grad_X: wp.array(dtype=T.Any, ndim=1),
    ):
        i = wp.tid()
        grad_X[i] = compute_grad_q(X[i], grad_output[i])
    return implement


def _make_kernel_2d(dtype):
    compute_grad_q = _make_compute_grad_q(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=2),
        grad_output: wp.array(dtype=T.Any, ndim=2),
        grad_X: wp.array(dtype=T.Any, ndim=2),
    ):
        i, j = wp.tid()  # type: ignore
        grad_X[i, j] = compute_grad_q(X[i, j], grad_output[i, j])
    return implement


def _make_kernel_3d(dtype):
    compute_grad_q = _make_compute_grad_q(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=3),
        grad_output: wp.array(dtype=T.Any, ndim=3),
        grad_X: wp.array(dtype=T.Any, ndim=3),
    ):
        i, j, k = wp.tid()  # type: ignore
        grad_X[i, j, k] = compute_grad_q(X[i, j, k], grad_output[i, j, k])
    return implement


def _make_kernel_4d(dtype):
    compute_grad_q = _make_compute_grad_q(dtype)

    @wp.kernel(enable_backward=False)
    def implement(
        X: wp.array(dtype=T.Any, ndim=4),
        grad_output: wp.array(dtype=T.Any, ndim=4),
        grad_X: wp.array(dtype=T.Any, ndim=4),
    ):
        i, j, k, l = wp.tid()  # type: ignore
        grad_X[i, j, k, l] = compute_grad_q(X[i, j, k, l], grad_output[i, j, k, l])
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

def SO3_Mat_bwd(
    X: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for SO3_Mat.
    
    Args:
        X: Quaternion tensor of shape (..., 4)
        grad_output: Gradient w.r.t. output matrix, shape (..., 3, 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 4)
    """
    from ...common.kernel_utils import TORCH_TO_WP_SCALAR
    
    # Prepare batch dimensions
    X, batch_info = prepare_batch_single(X)
    
    if batch_info.squeeze_output:
        grad_output = grad_output.unsqueeze(0)
    
    dtype = X.dtype
    device = X.device
    quat_type = wp_quat(dtype)
    mat33_type = wp_mat33(dtype)
    wp_scalar = TORCH_TO_WP_SCALAR[dtype]
    
    # Detach and ensure tensors are contiguous
    X = X.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=quat_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=mat33_type)
    
    # Allocate output gradient
    grad_X_tensor = torch.empty((*batch_info.shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    
    # Get kernel and launch
    kernel = KernelRegistry.get(_kernel_factories, batch_info.ndim, wp_scalar)
    wp.launch(
        kernel=kernel,
        dim=batch_info.shape,
        device=X_wp.device,
        inputs=[X_wp, grad_output_wp, grad_X_wp],
    )
    
    return finalize_output(grad_X_tensor, batch_info)
