import torch
from .fwd import SO3_Inv_fwd
from .bwd import SO3_Inv_bwd


class SO3_Inv(torch.autograd.Function):
    generate_vmap_rule = True
    
    @staticmethod
    def forward(ctx, X):
        out = SO3_Inv_fwd(X)
        ctx.save_for_backward(out.tensor())
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        return SO3_Inv_bwd(out, grad_output)

