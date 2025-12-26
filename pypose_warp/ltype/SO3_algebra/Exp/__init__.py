import torch
import pypose as pp
from .fwd import so3_Exp_fwd
from .bwd import so3_Exp_bwd


class so3_Exp(torch.autograd.Function):
    """
    Exponential map from so3 (Lie algebra) to SO3 (Lie group).
    
    Maps a 3D axis-angle vector to a unit quaternion.
    """
    generate_vmap_rule = True
    
    @staticmethod
    def forward(ctx, x):
        # Convert LieTensor to tensor if needed (for proper autograd tracking)
        x_tensor = x.tensor() if hasattr(x, 'tensor') else x
        
        # Create LieTensor for the forward function
        from ... import warpso3_type  # lazy import to avoid circular import
        x_lie = pp.LieTensor(x_tensor, ltype=warpso3_type) if not hasattr(x, 'ltype') else x
        
        out = so3_Exp_fwd(x_lie)
        
        # Save input tensor for backward (needed for Jacobian computation)
        ctx.save_for_backward(x_tensor)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # grad_output may be a LieTensor, extract tensor if needed
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        return so3_Exp_bwd(x, grad_output)
