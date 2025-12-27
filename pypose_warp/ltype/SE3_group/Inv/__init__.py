import torch
import pypose as pp

from .fwd import SE3_Inv_fwd
from .bwd import SE3_Inv_bwd


class SE3_Inv(torch.autograd.Function):
    """
    Autograd wrapper for SE3_Inv forward and backward.
    
    SE3_Inv computes the inverse of an SE3 transformation:
        q_inv = conjugate(q) = (-q_x, -q_y, -q_z, q_w)
        t_inv = -R_inv @ t
        Y = (t_inv, q_inv)
    
    This is equivalent to the matrix inverse of the SE3 4x4 matrix.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(X: pp.LieTensor) -> torch.Tensor:
        # Run forward pass
        out = SE3_Inv_fwd(X)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Save the output (inverse) for backward - we don't need X
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        Y, = ctx.saved_tensors
        
        # Compute gradient
        grad_X = SE3_Inv_bwd(Y, grad_output)
        
        return grad_X

