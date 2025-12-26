# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T

from ....utils.warp_utils import wp_quat_type, wp_mat33_type


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
# NOTE: This is equivalent to the standard formula for unit quaternions,
# but has different derivatives because the formula is written differently.
#
# Derivatives:
#   dR[0,0]/dx = 4*x,  dR[1,1]/dy = 4*y,  dR[2,2]/dz = 4*z
#   dR[0,0]/dw = 4*w,  dR[1,1]/dw = 4*w,  dR[2,2]/dw = 4*w
#   dR[0,1]/dx = 2*y,  dR[0,1]/dy = 2*x,  dR[0,1]/dz = -2*w, dR[0,1]/dw = -2*z
#   dR[0,2]/dx = 2*z,  dR[0,2]/dy = 2*w,  dR[0,2]/dz = 2*x,  dR[0,2]/dw = 2*y
#   dR[1,0]/dx = 2*y,  dR[1,0]/dy = 2*x,  dR[1,0]/dz = 2*w,  dR[1,0]/dw = 2*z
#   dR[1,2]/dx = -2*w, dR[1,2]/dy = 2*z,  dR[1,2]/dz = 2*y,  dR[1,2]/dw = -2*x
#   dR[2,0]/dx = 2*z,  dR[2,0]/dy = -2*w, dR[2,0]/dz = 2*x,  dR[2,0]/dw = -2*y
#   dR[2,1]/dx = 2*w,  dR[2,1]/dy = 2*z,  dR[2,1]/dz = 2*y,  dR[2,1]/dw = 2*x
#
# Given grad_R (gradient w.r.t. matrix elements), compute grad_q:
#   grad_x = 4*x*G00 + 2*y*(G01 + G10) + 2*z*(G02 + G20) + 2*w*(G21 - G12)
#   grad_y = 2*x*(G01 + G10) + 4*y*G11 + 2*z*(G12 + G21) + 2*w*(G02 - G20)
#   grad_z = 2*x*(G02 + G20) + 2*y*(G12 + G21) + 4*z*G22 + 2*w*(G10 - G01)
#   grad_w = 4*w*(G00 + G11 + G22) + 2*x*(G21 - G12) + 2*y*(G02 - G20) + 2*z*(G10 - G01)
#
# where G = grad_R (Gij = grad_R[i,j])
# =============================================================================


@wp.func
def compute_grad_q(q: wp.quatf, G: wp.mat33f) -> wp.quatf:
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
    
    # Correct formulas based on Warp's quat_to_matrix using quat_rotate
    grad_x = wp.float32(4.0) * x * G[0, 0] + wp.float32(2.0) * y * G01_plus_G10 + wp.float32(2.0) * z * G02_plus_G20 + wp.float32(2.0) * w * G21_minus_G12
    grad_y = wp.float32(2.0) * x * G01_plus_G10 + wp.float32(4.0) * y * G[1, 1] + wp.float32(2.0) * z * G12_plus_G21 + wp.float32(2.0) * w * G02_minus_G20
    grad_z = wp.float32(2.0) * x * G02_plus_G20 + wp.float32(2.0) * y * G12_plus_G21 + wp.float32(4.0) * z * G[2, 2] + wp.float32(2.0) * w * G10_minus_G01
    grad_w = wp.float32(4.0) * w * (G[0, 0] + G[1, 1] + G[2, 2]) + wp.float32(2.0) * x * G21_minus_G12 + wp.float32(2.0) * y * G02_minus_G20 + wp.float32(2.0) * z * G10_minus_G01
    
    return wp.quatf(grad_x, grad_y, grad_z, grad_w)


@wp.func
def compute_grad_q(q: wp.quatd, G: wp.mat33d) -> wp.quatd:
    """Compute quaternion gradient from rotation matrix gradient (float64)."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    
    G01_plus_G10 = G[0, 1] + G[1, 0]
    G02_plus_G20 = G[0, 2] + G[2, 0]
    G12_plus_G21 = G[1, 2] + G[2, 1]
    
    G21_minus_G12 = G[2, 1] - G[1, 2]
    G02_minus_G20 = G[0, 2] - G[2, 0]
    G10_minus_G01 = G[1, 0] - G[0, 1]
    
    # Correct formulas based on Warp's quat_to_matrix using quat_rotate
    grad_x = wp.float64(4.0) * x * G[0, 0] + wp.float64(2.0) * y * G01_plus_G10 + wp.float64(2.0) * z * G02_plus_G20 + wp.float64(2.0) * w * G21_minus_G12
    grad_y = wp.float64(2.0) * x * G01_plus_G10 + wp.float64(4.0) * y * G[1, 1] + wp.float64(2.0) * z * G12_plus_G21 + wp.float64(2.0) * w * G02_minus_G20
    grad_z = wp.float64(2.0) * x * G02_plus_G20 + wp.float64(2.0) * y * G12_plus_G21 + wp.float64(4.0) * z * G[2, 2] + wp.float64(2.0) * w * G10_minus_G01
    grad_w = wp.float64(4.0) * w * (G[0, 0] + G[1, 1] + G[2, 2]) + wp.float64(2.0) * x * G21_minus_G12 + wp.float64(2.0) * y * G02_minus_G20 + wp.float64(2.0) * z * G10_minus_G01
    
    return wp.quatd(grad_x, grad_y, grad_z, grad_w)


@wp.func
def compute_grad_q(q: wp.quath, G: wp.mat33h) -> wp.quath:
    """Compute quaternion gradient from rotation matrix gradient (float16)."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    
    G01_plus_G10 = G[0, 1] + G[1, 0]
    G02_plus_G20 = G[0, 2] + G[2, 0]
    G12_plus_G21 = G[1, 2] + G[2, 1]
    
    G21_minus_G12 = G[2, 1] - G[1, 2]
    G02_minus_G20 = G[0, 2] - G[2, 0]
    G10_minus_G01 = G[1, 0] - G[0, 1]
    
    # Correct formulas based on Warp's quat_to_matrix using quat_rotate
    grad_x = wp.float16(4.0) * x * G[0, 0] + wp.float16(2.0) * y * G01_plus_G10 + wp.float16(2.0) * z * G02_plus_G20 + wp.float16(2.0) * w * G21_minus_G12
    grad_y = wp.float16(2.0) * x * G01_plus_G10 + wp.float16(4.0) * y * G[1, 1] + wp.float16(2.0) * z * G12_plus_G21 + wp.float16(2.0) * w * G02_minus_G20
    grad_z = wp.float16(2.0) * x * G02_plus_G20 + wp.float16(2.0) * y * G12_plus_G21 + wp.float16(4.0) * z * G[2, 2] + wp.float16(2.0) * w * G10_minus_G01
    grad_w = wp.float16(4.0) * w * (G[0, 0] + G[1, 1] + G[2, 2]) + wp.float16(2.0) * x * G21_minus_G12 + wp.float16(2.0) * y * G02_minus_G20 + wp.float16(2.0) * z * G10_minus_G01
    
    return wp.quath(grad_x, grad_y, grad_z, grad_w)


# =============================================================================
# Backward kernels for 1D to 4D batch dimensions
# =============================================================================

@wp.kernel(enable_backward=False)
def SO3_Mat_bwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    grad_output: wp.array(dtype=T.Any, ndim=1),
    grad_X: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    grad_X[i] = compute_grad_q(X[i], grad_output[i])


@wp.kernel(enable_backward=False)
def SO3_Mat_bwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    grad_output: wp.array(dtype=T.Any, ndim=2),
    grad_X: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid()  # type: ignore
    grad_X[i, j] = compute_grad_q(X[i, j], grad_output[i, j])


@wp.kernel(enable_backward=False)
def SO3_Mat_bwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    grad_output: wp.array(dtype=T.Any, ndim=3),
    grad_X: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid()  # type: ignore
    grad_X[i, j, k] = compute_grad_q(X[i, j, k], grad_output[i, j, k])


@wp.kernel(enable_backward=False)
def SO3_Mat_bwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    grad_output: wp.array(dtype=T.Any, ndim=4),
    grad_X: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid()  # type: ignore
    grad_X[i, j, k, l] = compute_grad_q(X[i, j, k, l], grad_output[i, j, k, l])


# =============================================================================
# Kernel selection map
# =============================================================================

_SO3_Mat_bwd_kernels = {
    1: SO3_Mat_bwd_kernel_1d,
    2: SO3_Mat_bwd_kernel_2d,
    3: SO3_Mat_bwd_kernel_3d,
    4: SO3_Mat_bwd_kernel_4d,
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
    batch_shape = X.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
        X = X.unsqueeze(0)
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
    mat33_type = wp_mat33_type(dtype)
    
    # Detach and ensure tensors are contiguous for warp conversion
    X = X.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X, dtype=quat_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=mat33_type)
    
    # Allocate output gradient
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    
    # Select and launch kernel
    kernel = _SO3_Mat_bwd_kernels[ndim]
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, grad_output_wp, grad_X_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
    
    return grad_X_tensor


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# =============================================================================

__SO3_Mat_bwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_Mat_bwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        grad_output=wp.array(dtype=wp.mat33h, ndim=1),
        grad_X=wp.array(dtype=wp.quath, ndim=1),
    )),
    wp.overload(SO3_Mat_bwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        grad_output=wp.array(dtype=wp.mat33f, ndim=1),
        grad_X=wp.array(dtype=wp.quatf, ndim=1),
    )),
    wp.overload(SO3_Mat_bwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        grad_output=wp.array(dtype=wp.mat33d, ndim=1),
        grad_X=wp.array(dtype=wp.quatd, ndim=1),
    )),
    
    # 2D kernels
    wp.overload(SO3_Mat_bwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        grad_output=wp.array(dtype=wp.mat33h, ndim=2),
        grad_X=wp.array(dtype=wp.quath, ndim=2),
    )),
    wp.overload(SO3_Mat_bwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        grad_output=wp.array(dtype=wp.mat33f, ndim=2),
        grad_X=wp.array(dtype=wp.quatf, ndim=2),
    )),
    wp.overload(SO3_Mat_bwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        grad_output=wp.array(dtype=wp.mat33d, ndim=2),
        grad_X=wp.array(dtype=wp.quatd, ndim=2),
    )),
    
    # 3D kernels
    wp.overload(SO3_Mat_bwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        grad_output=wp.array(dtype=wp.mat33h, ndim=3),
        grad_X=wp.array(dtype=wp.quath, ndim=3),
    )),
    wp.overload(SO3_Mat_bwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        grad_output=wp.array(dtype=wp.mat33f, ndim=3),
        grad_X=wp.array(dtype=wp.quatf, ndim=3),
    )),
    wp.overload(SO3_Mat_bwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        grad_output=wp.array(dtype=wp.mat33d, ndim=3),
        grad_X=wp.array(dtype=wp.quatd, ndim=3),
    )),
    
    # 4D kernels
    wp.overload(SO3_Mat_bwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        grad_output=wp.array(dtype=wp.mat33h, ndim=4),
        grad_X=wp.array(dtype=wp.quath, ndim=4),
    )),
    wp.overload(SO3_Mat_bwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        grad_output=wp.array(dtype=wp.mat33f, ndim=4),
        grad_X=wp.array(dtype=wp.quatf, ndim=4),
    )),
    wp.overload(SO3_Mat_bwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        grad_output=wp.array(dtype=wp.mat33d, ndim=4),
        grad_X=wp.array(dtype=wp.quatd, ndim=4),
    )),
]

