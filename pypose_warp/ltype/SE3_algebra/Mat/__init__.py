"""
se3 Mat: Convert se3 twist to 4x4 transformation matrix.

This is equivalent to se3.Exp().matrix() but computed in a single fused kernel.
"""

import torch
import pypose as pp
from .fwd import se3_Mat_fwd
from .bwd import se3_Mat_bwd


class se3_Mat(torch.autograd.Function):
    """
    Convert se3 twist to 4x4 transformation matrix.
    
    This is equivalent to:
        se3.Exp().matrix()
    
    But computed in a single fused kernel for efficiency.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, x):
        out = se3_Mat_fwd(x)
        # Save forward input for backward
        ctx.save_for_backward(x.tensor() if hasattr(x, 'tensor') else x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return se3_Mat_bwd(x, grad_output)

