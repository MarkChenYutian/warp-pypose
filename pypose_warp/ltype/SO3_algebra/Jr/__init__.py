"""
so3 Right Jacobian (Jr) operator with Warp acceleration.

The right Jacobian Jr of so3 relates infinitesimal perturbations to the
exponential map:
    Exp(x + dx) ≈ Exp(x) @ Exp(Jr(x) @ dx)

Formula:
    Jr = I - (1 - cos(θ)) / θ² * K + (θ - sin(θ)) / θ³ * K @ K

where:
    K = skew(x)
    θ = ||x||
"""

import torch
import pypose as pp
import warnings
from .fwd import so3_Jr_fwd
from .bwd import so3_Jr_bwd


class so3_Jr(torch.autograd.Function):
    """
    Compute the right Jacobian Jr of so3.
    
    The forward pass uses an optimized Warp kernel.
    The backward pass uses PyTorch autograd composition.
    """
    generate_vmap_rule = True
    
    @staticmethod
    def forward(ctx, x: pp.LieTensor) -> torch.Tensor:
        """
        Compute the right Jacobian Jr.
        
        Args:
            x: so3 LieTensor of shape (..., 3)
            
        Returns:
            Right Jacobian tensor of shape (..., 3, 3)
        """
        if x.dtype == torch.float16:
            warnings.warn("[PRECISION] so3_Jr operator is potentially numeric instable with fp16. Nan may occur unexpectedly.")
        
        # Save input tensor for backward
        ctx.save_for_backward(x.tensor())
        return so3_Jr_fwd(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Compute the gradient of the loss w.r.t. the input x.
        
        Args:
            grad_output: Gradient of loss w.r.t. Jr output, shape (..., 3, 3)
            
        Returns:
            Gradient of loss w.r.t. x, shape (..., 3)
        """
        x_tensor, = ctx.saved_tensors
        grad_x = so3_Jr_bwd(x_tensor, grad_output)
        return grad_x


# Export the forward and backward functions for direct use
so3_Jr_fwd = so3_Jr_fwd
so3_Jr_bwd = so3_Jr_bwd

