import torch
import pypose as pp
from .fwd import SO3_Mat_fwd
from .bwd import SO3_Mat_bwd


class SO3_Mat(torch.autograd.Function):
    """
    Convert SO3 quaternion to 3x3 rotation matrix.
    
    This is more efficient than PyPose's default implementation which uses:
        I = eye(3); return X.Act(I).transpose(-1,-2)
    
    By directly computing the rotation matrix from quaternion components,
    we avoid the overhead of creating identity matrix and applying rotations.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X):
        out = SO3_Mat_fwd(X)
        # Save quaternion for backward
        X_tensor = X.tensor() if isinstance(X, pp.LieTensor) else X
        ctx.save_for_backward(X_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X_tensor, = ctx.saved_tensors
        grad_X = SO3_Mat_bwd(X_tensor, grad_output)
        return grad_X

