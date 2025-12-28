import torch
import pypose as pp

from .fwd import SE3_AdjTXa_fwd
from .bwd import SE3_AdjTXa_bwd


class SE3_AdjTXa(torch.autograd.Function):
    """
    SE3 transpose adjoint action on se3 Lie algebra element.
    
    Forward: out = Adj^T(X) @ a = Adj(X^{-1}) @ a
        where Adj^T(X) is the transpose of the 6x6 adjoint matrix:
            Adj^T = [R^T,              0   ]
                    [-R^T @ skew(t),   R^T ]
    
    Backward:
        a_grad = Adj(X) @ grad
        X_grad[:6] = -a @ se3_adj(a_grad), X_grad[6] = 0
    """
    
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X, a):
        X_tensor = X.tensor() if hasattr(X, 'tensor') else X
        a_tensor = a.tensor() if hasattr(a, 'tensor') else a
        
        # Get batch shapes
        X_batch_shape = X_tensor.shape[:-1]
        a_batch_shape = a_tensor.shape[:-1]
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, a_batch_shape)
        
        # Expand tensors to broadcast shape
        X_expanded = X_tensor.expand(*out_batch_shape, 7)
        a_expanded = a_tensor.expand(*out_batch_shape, 6)
        
        # Call forward
        from .. import warpSE3_type
        out = SE3_AdjTXa_fwd(
            pp.LieTensor(X_expanded, ltype=warpSE3_type),
            pp.LieTensor(a_expanded, ltype=pp.se3_type)
        )
        
        # Save for backward
        ctx.save_for_backward(X_expanded.contiguous(), a_expanded.contiguous())
        ctx.X_batch_shape = X_batch_shape
        ctx.a_batch_shape = a_batch_shape
        ctx.out_batch_shape = out_batch_shape
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X_expanded, a_expanded = ctx.saved_tensors
        X_batch_shape = ctx.X_batch_shape
        a_batch_shape = ctx.a_batch_shape
        out_batch_shape = ctx.out_batch_shape
        
        # Compute gradients
        grad_X, grad_a = SE3_AdjTXa_bwd(X_expanded, a_expanded, grad_output)
        
        # Handle broadcasting reduction
        ndim_out = len(out_batch_shape)
        ndim_X = len(X_batch_shape)
        ndim_a = len(a_batch_shape)
        
        # Pad shapes to match output shape
        X_padded = (1,) * (ndim_out - ndim_X) + tuple(X_batch_shape)
        a_padded = (1,) * (ndim_out - ndim_a) + tuple(a_batch_shape)
        
        # Find dimensions to reduce for X
        reduce_dims_X = []
        for i, (x_dim, out_dim) in enumerate(zip(X_padded, out_batch_shape)):
            if x_dim == 1 and out_dim > 1:
                reduce_dims_X.append(i)
        
        # Find dimensions to reduce for a
        reduce_dims_a = []
        for i, (a_dim, out_dim) in enumerate(zip(a_padded, out_batch_shape)):
            if a_dim == 1 and out_dim > 1:
                reduce_dims_a.append(i)
        
        # Reduce gradients if necessary
        if reduce_dims_X:
            grad_X = grad_X.sum(dim=reduce_dims_X, keepdim=True)
        grad_X = grad_X.view(*X_batch_shape, 7)
        
        if reduce_dims_a:
            grad_a = grad_a.sum(dim=reduce_dims_a, keepdim=True)
        grad_a = grad_a.view(*a_batch_shape, 6)
        
        return grad_X, grad_a

