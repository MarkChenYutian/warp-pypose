import torch
from .fwd import SO3_Act_fwd
from .bwd import SO3_Act_bwd


class SO3_Act(torch.autograd.Function):
    generate_vmap_rule = True
    
    @staticmethod
    def forward(X, p):
        return SO3_Act_fwd(X, p)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        X, p = inputs
        out = output
        ctx.save_for_backward(X, out)
    
    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        return SO3_Act_bwd(X, out, grad_output)
