"""
SE3 Mat: Convert SE3 pose to 4x4 transformation matrix.

This module provides an efficient conversion from SE3 (7-element pose representation)
to 4x4 homogeneous transformation matrix.

SE3 representation: [tx, ty, tz, qx, qy, qz, qw]

4x4 Transformation matrix:
    [ R  | t ]     [ R00 R01 R02 | tx ]
    [----+---]  =  [ R10 R11 R12 | ty ]
    [ 0  | 1 ]     [ R20 R21 R22 | tz ]
                   [  0   0   0  | 1  ]

This is more efficient than PyPose's default implementation which uses:
    I = eye(4); return X.unsqueeze(-2).Act(I).transpose(-1,-2)
"""

import torch
import pypose as pp
from .fwd import SE3_Mat_fwd
from .bwd import SE3_Mat_bwd


class SE3_Mat(torch.autograd.Function):
    """
    Convert SE3 pose to 4x4 transformation matrix.
    
    This is more efficient than PyPose's default implementation which uses:
        I = eye(4); return X.unsqueeze(-2).Act(I).transpose(-1,-2)
    
    By directly constructing the matrix from translation and quaternion components,
    we avoid the overhead of creating identity matrix and applying transformations.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X):
        out = SE3_Mat_fwd(X)
        # Save SE3 pose for backward
        X_tensor = X.tensor() if isinstance(X, pp.LieTensor) else X
        ctx.save_for_backward(X_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X_tensor, = ctx.saved_tensors
        grad_X = SE3_Mat_bwd(X_tensor, grad_output)
        return grad_X

