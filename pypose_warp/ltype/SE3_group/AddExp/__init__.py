"""
SE3 AddExp: Fused Exp(delta) * X operation.

This module provides a fused kernel for the common operation:
    result = Exp(delta) * X

where delta is an se3 element (twist, shape (..., 6)) and X is an SE3 element
(pose, shape (..., 7)).

This is used in the add_ operation for SE3 LieTensors, which updates a pose
by a tangent space delta: X = Exp(delta) * X

The fused kernel is more efficient than separate Exp + Mul operations.
"""

import torch
import pypose as pp
from .fwd import SE3_AddExp_fwd
from .bwd import SE3_AddExp_bwd


class SE3_AddExp(torch.autograd.Function):
    """
    Fused Exp(delta) * X operation for SE3.
    
    Given:
    - delta: se3 twist vector of shape (..., 6)
    - X: SE3 pose of shape (..., 7)
    
    Computes:
    - Y = Exp(delta) * X (SE3 pose)
    
    This is a common operation in Lie group optimization where a pose
    is updated by a small delta in the tangent space.
    """
    generate_vmap_rule = True
    
    @staticmethod
    def forward(ctx, delta, X):
        """
        Forward pass: compute Exp(delta) * X.
        
        Args:
            delta: Tensor of shape (..., 6) - se3 twist [tau, phi]
            X: LieTensor of shape (..., 7) - SE3 pose [t, q]
            
        Returns:
            LieTensor of shape (..., 7) - result SE3 pose
        """
        # Convert LieTensor to tensor if needed
        delta_tensor = delta.tensor() if hasattr(delta, 'tensor') else delta
        
        out = SE3_AddExp_fwd(delta_tensor, X)
        
        # Save delta for backward (needed for Jacobian computation)
        ctx.save_for_backward(delta_tensor)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients w.r.t. delta and X.
        
        Args:
            grad_output: Gradient w.r.t. output SE3 pose
            
        Returns:
            Tuple of (grad_delta, grad_X)
        """
        delta, = ctx.saved_tensors
        
        # grad_output may be a LieTensor, extract tensor if needed
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        
        grad_delta, grad_X = SE3_AddExp_bwd(delta, grad_output)
        
        return grad_delta, grad_X

