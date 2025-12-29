"""
SO3 AddExp: Fused Exp(delta) * X operation.

This module provides a fused kernel for the common operation:
    result = Exp(delta) * X

where delta is an so3 element (axis-angle, shape (..., 3)) and X is an SO3 element
(quaternion, shape (..., 4)).

This is used in the add_ operation for SO3 LieTensors, which updates a rotation
by a tangent space delta: X = Exp(delta) * X

The fused kernel is more efficient than separate Exp + Mul operations.
"""

import torch
import pypose as pp
from .fwd import SO3_AddExp_fwd
from .bwd import SO3_AddExp_bwd


class SO3_AddExp(torch.autograd.Function):
    """
    Fused Exp(delta) * X operation for SO3.
    
    Given:
    - delta: so3 axis-angle vector of shape (..., 3)
    - X: SO3 quaternion of shape (..., 4)
    
    Computes:
    - Y = Exp(delta) * X (quaternion)
    
    This is a common operation in Lie group optimization where a rotation
    is updated by a small delta in the tangent space.
    """
    generate_vmap_rule = True
    
    @staticmethod
    def forward(ctx, delta, X):
        """
        Forward pass: compute Exp(delta) * X.
        
        Args:
            delta: Tensor of shape (..., 3) - axis-angle tangent vector
            X: LieTensor of shape (..., 4) - SO3 quaternion
            
        Returns:
            LieTensor of shape (..., 4) - result SO3 quaternion
        """
        # Convert LieTensor to tensor if needed
        delta_tensor = delta.tensor() if hasattr(delta, 'tensor') else delta
        
        out = SO3_AddExp_fwd(delta_tensor, X)
        
        # Save delta for backward (needed for Jacobian computation)
        ctx.save_for_backward(delta_tensor)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients w.r.t. delta and X.
        
        Args:
            grad_output: Gradient w.r.t. output quaternion
            
        Returns:
            Tuple of (grad_delta, grad_X)
        """
        delta, = ctx.saved_tensors
        
        # grad_output may be a LieTensor, extract tensor if needed
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        
        grad_delta, grad_X = SO3_AddExp_bwd(delta, grad_output)
        
        return grad_delta, grad_X

