import torch
import pypose as pp
from .fwd import SO3_AdjTXa_fwd
from .bwd import SO3_AdjTXa_bwd


class SO3_AdjTXa(torch.autograd.Function):
    """
    Autograd function for SO3 adjoint transpose action on Lie algebra elements.
    
    Computes: out = SO3_Adj(X^{-1}) @ a = R^T @ a
    where R is the 3x3 rotation matrix corresponding to quaternion X.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X: pp.LieTensor, a: torch.Tensor) -> torch.Tensor:
        out = SO3_AdjTXa_fwd(X, a)
        
        # Save original batch shapes for gradient reduction in backward
        X_batch_shape = X.shape[:-1]
        a_batch_shape = a.shape[:-1]
        out_batch_shape = out.shape[:-1]
        
        # Expand X and a to broadcast shape for backward computation
        X_expanded = X.tensor().expand(*out_batch_shape, 4)
        a_expanded = a.expand(*out_batch_shape, 3)
        
        ctx.save_for_backward(X_expanded, a_expanded)
        ctx.X_batch_shape = X_batch_shape
        ctx.a_batch_shape = a_batch_shape
        ctx.out_batch_shape = out_batch_shape
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X_expanded, a_expanded = ctx.saved_tensors
        X_batch_shape = ctx.X_batch_shape
        a_batch_shape = ctx.a_batch_shape
        out_batch_shape = ctx.out_batch_shape
        
        # Compute gradients at broadcast shape
        grad_X, grad_a = SO3_AdjTXa_bwd(X_expanded, a_expanded, grad_output)
        
        # Reduce gradients to original shapes by summing over broadcast dims
        ndim_out = len(out_batch_shape)
        ndim_X = len(X_batch_shape)
        ndim_a = len(a_batch_shape)
        
        # Pad shapes to same length for comparison
        X_padded = (1,) * (ndim_out - ndim_X) + tuple(X_batch_shape)
        a_padded = (1,) * (ndim_out - ndim_a) + tuple(a_batch_shape)
        
        # Find dims to reduce for X
        reduce_dims_X = []
        for i, (x_dim, out_dim) in enumerate(zip(X_padded, out_batch_shape)):
            if x_dim == 1 and out_dim > 1:
                reduce_dims_X.append(i)
        
        # Find dims to reduce for a
        reduce_dims_a = []
        for i, (a_dim, out_dim) in enumerate(zip(a_padded, out_batch_shape)):
            if a_dim == 1 and out_dim > 1:
                reduce_dims_a.append(i)
        
        # Reduce and reshape
        if reduce_dims_X:
            grad_X = grad_X.sum(dim=reduce_dims_X, keepdim=True)
        grad_X = grad_X.view(*X_batch_shape, 4)
        
        if reduce_dims_a:
            grad_a = grad_a.sum(dim=reduce_dims_a, keepdim=True)
        grad_a = grad_a.view(*a_batch_shape, 3)
        
        return grad_X, grad_a

