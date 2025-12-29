"""
se3 Exp: Exponential map from se3 Lie algebra to SE3 Lie group.

Converts se3 twist [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z] (6 elements)
to SE3 pose [tx, ty, tz, qx, qy, qz, qw] (7 elements).

Algorithm:
1. Extract tau (translation) and phi (rotation) from se3
2. Compute Jl = so3_Jl(phi) - left Jacobian
3. Compute t = Jl @ tau - translation in SE3
4. Compute q = so3_Exp(phi) - quaternion from axis-angle
5. Return [t, q]
"""

import torch
import pypose as pp
from .fwd import se3_Exp_fwd
from .bwd import se3_Exp_bwd


class se3_Exp(torch.autograd.Function):
    """
    Exponential map from se3 Lie algebra to SE3 Lie group.
    
    Maps se3 twist vector to SE3 pose representation.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, x):
        out = se3_Exp_fwd(x)
        # Save forward input for backward
        ctx.save_for_backward(x.tensor() if hasattr(x, 'tensor') else x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_output may be a LieTensor, extract tensor if needed
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        return se3_Exp_bwd(input, grad_output)

