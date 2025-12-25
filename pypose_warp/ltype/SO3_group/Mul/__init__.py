import torch
import pypose as pp
from .fwd import SO3_Mul_fwd
from .bwd import SO3_Mul_bwd


class SO3_Mul(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X, Y):
        out = SO3_Mul_fwd(X, Y)
        # Save original batch shapes for gradient reduction in backward
        X_batch_shape = X.shape[:-1]
        Y_batch_shape = Y.shape[:-1]
        out_batch_shape = out.shape[:-1]
        
        # Expand X to broadcast shape for backward computation
        X_expanded = X.tensor().expand(*out_batch_shape, 4)
        ctx.save_for_backward(X_expanded)
        ctx.X_batch_shape = X_batch_shape
        ctx.Y_batch_shape = Y_batch_shape
        ctx.out_batch_shape = out_batch_shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X_expanded, = ctx.saved_tensors
        X_batch_shape = ctx.X_batch_shape
        Y_batch_shape = ctx.Y_batch_shape
        out_batch_shape = ctx.out_batch_shape
        
        # Compute gradients at broadcast shape
        grad_X, grad_Y = SO3_Mul_bwd(X_expanded, grad_output)
        
        # Reduce gradients to original shapes by summing over broadcast dims
        # For X: sum over dims where X was broadcast (i.e., X had size 1)
        # For Y: sum over dims where Y was broadcast (i.e., Y had size 1)
        
        ndim_out = len(out_batch_shape)
        ndim_X = len(X_batch_shape)
        ndim_Y = len(Y_batch_shape)
        
        # Pad shapes to same length for comparison
        X_padded = (1,) * (ndim_out - ndim_X) + tuple(X_batch_shape)
        Y_padded = (1,) * (ndim_out - ndim_Y) + tuple(Y_batch_shape)
        
        # Find dims to reduce for X
        reduce_dims_X = []
        for i, (x_dim, out_dim) in enumerate(zip(X_padded, out_batch_shape)):
            if x_dim == 1 and out_dim > 1:
                reduce_dims_X.append(i)
        
        # Find dims to reduce for Y
        reduce_dims_Y = []
        for i, (y_dim, out_dim) in enumerate(zip(Y_padded, out_batch_shape)):
            if y_dim == 1 and out_dim > 1:
                reduce_dims_Y.append(i)
        
        # Reduce and reshape
        if reduce_dims_X:
            grad_X = grad_X.sum(dim=reduce_dims_X, keepdim=True)
        grad_X = grad_X.view(*X_batch_shape, 4)
        
        if reduce_dims_Y:
            grad_Y = grad_Y.sum(dim=reduce_dims_Y, keepdim=True)
        grad_Y = grad_Y.view(*Y_batch_shape, 4)
        
        return grad_X, grad_Y
