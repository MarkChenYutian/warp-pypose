"""
SE3 Log: Logarithm map from SE3 Lie group to se3 Lie algebra.

Converts SE3 pose [tx, ty, tz, qx, qy, qz, qw] (7 elements)
to se3 twist [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z] (6 elements).

Algorithm:
1. Extract quaternion q from SE3
2. Compute phi = SO3_Log(q) - rotation axis-angle
3. Compute Jl_inv = so3_Jl_inv(phi) - left Jacobian inverse  
4. Compute tau = Jl_inv @ translation
5. Return [tau, phi]
"""

import torch
import pypose as pp
from .fwd import SE3_Log_fwd
from .bwd import SE3_Log_bwd


class SE3_Log(torch.autograd.Function):
    """
    Logarithm map from SE3 Lie group to se3 Lie algebra.
    
    Maps SE3 pose representation to se3 twist vector.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X):
        out = SE3_Log_fwd(X)
        # Save forward output for backward
        ctx.save_for_backward(out.tensor())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # grad_output may be a LieTensor, extract tensor if needed
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        return SE3_Log_bwd(output, grad_output)

