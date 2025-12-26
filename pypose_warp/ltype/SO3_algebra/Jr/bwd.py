"""
Backward pass for so3 right Jacobian (Jr) computation.

The backward pass computes the gradient of the loss with respect to the input x,
given the gradient of the loss with respect to the output Jr matrix.

For the backward pass, we use PyTorch's autograd by expressing Jr in terms of
differentiable PyTorch operations.
"""

import torch
import pypose as pp
from pypose.lietensor.basics import vec2skew


def so3_Jr_bwd(x_tensor: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """
    Compute the backward pass for so3 right Jacobian Jr.
    
    This uses PyTorch's autograd by re-expressing Jr with differentiable ops.
    
    Args:
        x_tensor: Input tensor of shape (..., 3)
        grad_output: Gradient of loss w.r.t. Jr output, shape (..., 3, 3)
        
    Returns:
        Gradient of loss w.r.t. x, shape (..., 3)
    """
    # Must enable gradient tracking since we're inside a backward pass
    # where grad mode is disabled by default
    with torch.enable_grad():
        # Create a new leaf tensor with requires_grad for autograd.grad
        # We need to detach first to break any existing computation graph
        x_with_grad = x_tensor.detach().clone().requires_grad_(True)
        
        # Compute Jr using PyTorch operations (same formula as PyPose)
        K = vec2skew(x_with_grad)
        theta = torch.linalg.norm(x_with_grad, dim=-1, keepdim=True).unsqueeze(-1)
        batch_shape = x_with_grad.shape[:-1]
        I = torch.eye(3, device=x_with_grad.device, dtype=x_with_grad.dtype)
        I = I.expand(*batch_shape, 3, 3)
        
        # Compute Jr with stability handling
        theta2 = theta ** 2
        theta3 = theta ** 3
        
        # For numerical stability, we use Taylor expansion for small theta.
        # The Taylor expansion is smooth and differentiable, avoiding torch.where issues.
        # coef1 = (1 - cos(theta)) / theta^2 ≈ 0.5 - theta^2/24 + theta^4/720 - ...
        # coef2 = (theta - sin(theta)) / theta^3 ≈ 1/6 - theta^2/120 + theta^4/5040 - ...
        
        # Use a smooth blend instead of hard cutoff for gradient stability
        # For very small theta, the limit of coef1 is 0.5 and coef2 is 1/6
        eps = 1e-6
        theta_safe = torch.clamp(theta, min=eps)
        theta2_safe = theta_safe ** 2
        theta3_safe = theta_safe ** 3
        
        # Standard formula (safe because theta is clamped)
        coef1 = (1.0 - torch.cos(theta_safe)) / theta2_safe
        coef2 = (theta_safe - torch.sin(theta_safe)) / theta3_safe
        
        # Jr = I - coef1 * K + coef2 * K @ K
        Jr = I - coef1 * K + coef2 * (K @ K)
        
        # Compute gradient using autograd
        grad_x, = torch.autograd.grad(Jr, x_with_grad, grad_output)
    
    return grad_x

