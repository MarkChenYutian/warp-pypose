import torch
import pypose as pp

from .fwd import SE3_Act4_fwd
from .bwd import SE3_Act4_bwd


class SE3_Act4(torch.autograd.Function):
    """
    Autograd wrapper for SE3_Act4 forward and backward.
    
    SE3_Act4 applies an SE3 transformation to 4D homogeneous points:
        out[:3] = R @ p[:3] + t * p[3]
        out[3] = p[3]
    
    This is equivalent to multiplying by the SE3 4x4 matrix:
        [R | t] [p[:3]]   [R @ p[:3] + t * p[3]]
        [0 | 1] [p[3] ] = [p[3]                ]
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
        # Run forward pass
        out = SE3_Act4_fwd(X, p)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, p = inputs
        X_tensor = X.tensor()
        
        # Get batch shapes for gradient reduction in backward
        X_batch_shape = X_tensor.shape[:-1]
        p_batch_shape = p.shape[:-1]
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
        
        # Expand X to broadcast shape for backward
        X_expanded = X_tensor.expand(*out_batch_shape, 7)
        
        # Save tensors for backward
        ctx.save_for_backward(X_expanded, output)
        ctx.X_batch_shape = X_batch_shape
        ctx.p_batch_shape = p_batch_shape
        ctx.out_batch_shape = out_batch_shape

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X_expanded, out = ctx.saved_tensors
        X_batch_shape = ctx.X_batch_shape
        p_batch_shape = ctx.p_batch_shape
        out_batch_shape = ctx.out_batch_shape
        
        # Compute gradients at the broadcasted shape
        grad_X, grad_p = SE3_Act4_bwd(X_expanded, out, grad_output)
        
        # Reduce gradients to original input shapes by summing over broadcasted dims
        # For X:
        if X_batch_shape != out_batch_shape:
            X_batch_ndim = len(X_batch_shape)
            out_batch_ndim = len(out_batch_shape)
            
            # Prepend 1s to match out_batch_ndim
            X_batch_padded = (1,) * (out_batch_ndim - X_batch_ndim) + tuple(X_batch_shape)
            
            # Sum over broadcasted dimensions
            reduce_dims = []
            for i, (x_size, o_size) in enumerate(zip(X_batch_padded, out_batch_shape)):
                if x_size == 1 and o_size != 1:
                    reduce_dims.append(i)
            
            if reduce_dims:
                grad_X = grad_X.sum(dim=reduce_dims, keepdim=True)
            
            # Reshape back to original X_batch_shape
            grad_X = grad_X.view(*X_batch_shape, 7)
        
        # For p:
        if p_batch_shape != out_batch_shape:
            p_batch_ndim = len(p_batch_shape)
            out_batch_ndim = len(out_batch_shape)
            
            p_batch_padded = (1,) * (out_batch_ndim - p_batch_ndim) + tuple(p_batch_shape)
            
            reduce_dims = []
            for i, (p_size, o_size) in enumerate(zip(p_batch_padded, out_batch_shape)):
                if p_size == 1 and o_size != 1:
                    reduce_dims.append(i)
            
            if reduce_dims:
                grad_p = grad_p.sum(dim=reduce_dims, keepdim=True)
            
            grad_p = grad_p.view(*p_batch_shape, 4)
        
        return grad_X, grad_p

