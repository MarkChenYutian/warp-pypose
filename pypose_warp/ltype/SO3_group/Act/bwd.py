# pyright: reportInvalidTypeForm=false
# NOTE: warp language's type annotation spec does not match Pyright spec completely.

import torch
import warp as wp
import typing as T
import pypose as pp

from ....utils.warp_utils import wp_quat_type, wp_vec3_type


# =============================================================================
# Backward kernels for SO3_Act
# 
# Given: grad_output (gradient w.r.t. output of forward pass)
# Compute: 
#   - X_grad: gradient w.r.t. quaternion X (shape ..., 4)
#   - p_grad: gradient w.r.t. point p (shape ..., 3)
#
# From PyPose:
#   X_grad[:3] = grad_output @ skew(-out)  (Jacobian w.r.t. Lie algebra)
#   X_grad[3] = 0  (w component has zero gradient)
#   p_grad = grad_output @ R  (where R = rotation matrix from X)
# =============================================================================

@wp.func
def compute_grad_X(o: wp.vec3f, g: wp.vec3f) -> wp.quatf:
    """Compute quaternion gradient: (cross(out, grad), 0)"""
    c = wp.cross(o, g)
    return wp.quatf(c[0], c[1], c[2], wp.float32(0.0))

@wp.func
def compute_grad_X(o: wp.vec3d, g: wp.vec3d) -> wp.quatd:
    """Compute quaternion gradient: (cross(out, grad), 0)"""
    c = wp.cross(o, g)
    return wp.quatd(c[0], c[1], c[2], wp.float64(0.0))

@wp.func
def compute_grad_X(o: wp.vec3h, g: wp.vec3h) -> wp.quath:
    """Compute quaternion gradient: (cross(out, grad), 0)"""
    c = wp.cross(o, g)
    return wp.quath(c[0], c[1], c[2], wp.float16(0.0))


@wp.kernel(enable_backward=False)
def SO3_Act_bwd_kernel_1d(
    X: wp.array(dtype=T.Any, ndim=1),
    out: wp.array(dtype=T.Any, ndim=1),
    grad_output: wp.array(dtype=T.Any, ndim=1),
    grad_X: wp.array(dtype=T.Any, ndim=1),
    grad_p: wp.array(dtype=T.Any, ndim=1),
):
    i = wp.tid()
    
    q = X[i]
    o = out[i]
    g = grad_output[i]
    
    # Rotation matrix from quaternion
    R = wp.quat_to_matrix(q)
    
    # grad_p = grad_output @ R (row vector times matrix)
    # In warp, R @ g is matrix times column vector, so we need R^T @ g
    grad_p[i] = wp.transpose(R) @ g
    
    # X_grad = (cross(out, grad), 0) - quaternion with w=0
    grad_X[i] = compute_grad_X(o, g)


@wp.kernel(enable_backward=False)
def SO3_Act_bwd_kernel_2d(
    X: wp.array(dtype=T.Any, ndim=2),
    out: wp.array(dtype=T.Any, ndim=2),
    grad_output: wp.array(dtype=T.Any, ndim=2),
    grad_X: wp.array(dtype=T.Any, ndim=2),
    grad_p: wp.array(dtype=T.Any, ndim=2),
):
    i, j = wp.tid()  # type: ignore
    
    q = X[i, j]
    o = out[i, j]
    g = grad_output[i, j]
    
    R = wp.quat_to_matrix(q)
    grad_p[i, j] = wp.transpose(R) @ g
    grad_X[i, j] = compute_grad_X(o, g)


@wp.kernel(enable_backward=False)
def SO3_Act_bwd_kernel_3d(
    X: wp.array(dtype=T.Any, ndim=3),
    out: wp.array(dtype=T.Any, ndim=3),
    grad_output: wp.array(dtype=T.Any, ndim=3),
    grad_X: wp.array(dtype=T.Any, ndim=3),
    grad_p: wp.array(dtype=T.Any, ndim=3),
):
    i, j, k = wp.tid()  # type: ignore
    
    q = X[i, j, k]
    o = out[i, j, k]
    g = grad_output[i, j, k]
    
    R = wp.quat_to_matrix(q)
    grad_p[i, j, k] = wp.transpose(R) @ g
    grad_X[i, j, k] = compute_grad_X(o, g)


@wp.kernel(enable_backward=False)
def SO3_Act_bwd_kernel_4d(
    X: wp.array(dtype=T.Any, ndim=4),
    out: wp.array(dtype=T.Any, ndim=4),
    grad_output: wp.array(dtype=T.Any, ndim=4),
    grad_X: wp.array(dtype=T.Any, ndim=4),
    grad_p: wp.array(dtype=T.Any, ndim=4),
):
    i, j, k, l = wp.tid()  # type: ignore
    
    q = X[i, j, k, l]
    o = out[i, j, k, l]
    g = grad_output[i, j, k, l]
    
    R = wp.quat_to_matrix(q)
    grad_p[i, j, k, l] = wp.transpose(R) @ g
    grad_X[i, j, k, l] = compute_grad_X(o, g)


# =============================================================================
# Kernel selection map
# =============================================================================

_SO3_Act_bwd_kernels = {
    1: SO3_Act_bwd_kernel_1d,
    2: SO3_Act_bwd_kernel_2d,
    3: SO3_Act_bwd_kernel_3d,
    4: SO3_Act_bwd_kernel_4d,
}


# =============================================================================
# Main backward function
# =============================================================================

def SO3_Act_bwd(
    X: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SO3_Act.
    
    Args:
        X: Quaternion tensor of shape (..., 4) - expanded to broadcast shape
        out: Output from forward pass, shape (..., 3) (saved from forward)
        grad_output: Gradient w.r.t. output, shape (..., 3)
        
    Returns:
        grad_X: Gradient w.r.t. X, shape (..., 4)
        grad_p: Gradient w.r.t. p, shape (..., 3)
    """
    X_tensor = X
    batch_shape = X_tensor.shape[:-1]
    ndim = len(batch_shape)
    
    if ndim == 0:
        # Scalar case: add dummy batch dimension
        X_tensor = X_tensor.unsqueeze(0)
        out = out.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        batch_shape = (1,)
        ndim = 1
        squeeze_output = True
    else:
        squeeze_output = False
    
    if ndim > 4:
        raise NotImplementedError(f"Batch dimensions > 4 not supported. Got shape {batch_shape}")
    
    dtype = X_tensor.dtype
    device = X_tensor.device
    quat_type = wp_quat_type(dtype)
    vec3_type = wp_vec3_type(dtype)
    
    # Detach and ensure tensors are contiguous for warp conversion
    # (detach avoids warnings about accessing .grad on non-leaf tensors)
    X_tensor = X_tensor.detach().contiguous()
    out = out.detach().contiguous()
    grad_output = grad_output.detach().contiguous()
    
    # Convert to warp arrays
    X_wp = wp.from_torch(X_tensor, dtype=quat_type)
    out_wp = wp.from_torch(out, dtype=vec3_type)
    grad_output_wp = wp.from_torch(grad_output, dtype=vec3_type)
    
    # Allocate output gradients directly with correct shapes
    # grad_X is quaternion (4 components) with last component always 0
    grad_X_tensor = torch.empty((*batch_shape, 4), dtype=dtype, device=device)
    grad_p_tensor = torch.empty((*batch_shape, 3), dtype=dtype, device=device)
    
    grad_X_wp = wp.from_torch(grad_X_tensor, dtype=quat_type)
    grad_p_wp = wp.from_torch(grad_p_tensor, dtype=vec3_type)
    
    # Select and launch kernel
    kernel = _SO3_Act_bwd_kernels[ndim]
    wp.launch(
        kernel=kernel,
        dim=batch_shape,
        device=X_wp.device,
        inputs=[X_wp, out_wp, grad_output_wp, grad_X_wp, grad_p_wp],
    )
    
    if squeeze_output:
        grad_X_tensor = grad_X_tensor.squeeze(0)
        grad_p_tensor = grad_p_tensor.squeeze(0)
    
    return grad_X_tensor, grad_p_tensor


# =============================================================================
# Concrete kernel overloads for each precision and ndim combination
# =============================================================================

__SO3_Act_bwd_concrete_kernels = [
    # 1D kernels
    wp.overload(SO3_Act_bwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quath, ndim=1),
        out=wp.array(dtype=wp.vec3h, ndim=1),
        grad_output=wp.array(dtype=wp.vec3h, ndim=1),
        grad_X=wp.array(dtype=wp.quath, ndim=1),
        grad_p=wp.array(dtype=wp.vec3h, ndim=1),
    )),
    wp.overload(SO3_Act_bwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatf, ndim=1),
        out=wp.array(dtype=wp.vec3f, ndim=1),
        grad_output=wp.array(dtype=wp.vec3f, ndim=1),
        grad_X=wp.array(dtype=wp.quatf, ndim=1),
        grad_p=wp.array(dtype=wp.vec3f, ndim=1),
    )),
    wp.overload(SO3_Act_bwd_kernel_1d, dict(
        X=wp.array(dtype=wp.quatd, ndim=1),
        out=wp.array(dtype=wp.vec3d, ndim=1),
        grad_output=wp.array(dtype=wp.vec3d, ndim=1),
        grad_X=wp.array(dtype=wp.quatd, ndim=1),
        grad_p=wp.array(dtype=wp.vec3d, ndim=1),
    )),
    
    # 2D kernels
    wp.overload(SO3_Act_bwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quath, ndim=2),
        out=wp.array(dtype=wp.vec3h, ndim=2),
        grad_output=wp.array(dtype=wp.vec3h, ndim=2),
        grad_X=wp.array(dtype=wp.quath, ndim=2),
        grad_p=wp.array(dtype=wp.vec3h, ndim=2),
    )),
    wp.overload(SO3_Act_bwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatf, ndim=2),
        out=wp.array(dtype=wp.vec3f, ndim=2),
        grad_output=wp.array(dtype=wp.vec3f, ndim=2),
        grad_X=wp.array(dtype=wp.quatf, ndim=2),
        grad_p=wp.array(dtype=wp.vec3f, ndim=2),
    )),
    wp.overload(SO3_Act_bwd_kernel_2d, dict(
        X=wp.array(dtype=wp.quatd, ndim=2),
        out=wp.array(dtype=wp.vec3d, ndim=2),
        grad_output=wp.array(dtype=wp.vec3d, ndim=2),
        grad_X=wp.array(dtype=wp.quatd, ndim=2),
        grad_p=wp.array(dtype=wp.vec3d, ndim=2),
    )),
    
    # 3D kernels
    wp.overload(SO3_Act_bwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quath, ndim=3),
        out=wp.array(dtype=wp.vec3h, ndim=3),
        grad_output=wp.array(dtype=wp.vec3h, ndim=3),
        grad_X=wp.array(dtype=wp.quath, ndim=3),
        grad_p=wp.array(dtype=wp.vec3h, ndim=3),
    )),
    wp.overload(SO3_Act_bwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatf, ndim=3),
        out=wp.array(dtype=wp.vec3f, ndim=3),
        grad_output=wp.array(dtype=wp.vec3f, ndim=3),
        grad_X=wp.array(dtype=wp.quatf, ndim=3),
        grad_p=wp.array(dtype=wp.vec3f, ndim=3),
    )),
    wp.overload(SO3_Act_bwd_kernel_3d, dict(
        X=wp.array(dtype=wp.quatd, ndim=3),
        out=wp.array(dtype=wp.vec3d, ndim=3),
        grad_output=wp.array(dtype=wp.vec3d, ndim=3),
        grad_X=wp.array(dtype=wp.quatd, ndim=3),
        grad_p=wp.array(dtype=wp.vec3d, ndim=3),
    )),
    
    # 4D kernels
    wp.overload(SO3_Act_bwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quath, ndim=4),
        out=wp.array(dtype=wp.vec3h, ndim=4),
        grad_output=wp.array(dtype=wp.vec3h, ndim=4),
        grad_X=wp.array(dtype=wp.quath, ndim=4),
        grad_p=wp.array(dtype=wp.vec3h, ndim=4),
    )),
    wp.overload(SO3_Act_bwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatf, ndim=4),
        out=wp.array(dtype=wp.vec3f, ndim=4),
        grad_output=wp.array(dtype=wp.vec3f, ndim=4),
        grad_X=wp.array(dtype=wp.quatf, ndim=4),
        grad_p=wp.array(dtype=wp.vec3f, ndim=4),
    )),
    wp.overload(SO3_Act_bwd_kernel_4d, dict(
        X=wp.array(dtype=wp.quatd, ndim=4),
        out=wp.array(dtype=wp.vec3d, ndim=4),
        grad_output=wp.array(dtype=wp.vec3d, ndim=4),
        grad_X=wp.array(dtype=wp.quatd, ndim=4),
        grad_p=wp.array(dtype=wp.vec3d, ndim=4),
    )),
]
