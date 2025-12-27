import torch
import pypose as pp

from .fwd import SE3_Mul_fwd
from .bwd import SE3_Mul_bwd


class SE3_Mul(torch.autograd.Function):
    """
    SE3 group multiplication (composition) with custom backward.
    
    Forward: out = X @ Y
        t_out = t_X + R_X @ t_Y
        q_out = q_X * q_Y
    
    Backward:
        X_grad[:6] = grad_output[:6], X_grad[6] = 0
        Y_grad[:6] = grad_output[:6] @ SE3_Adj(X), Y_grad[6] = 0
    """
    
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X, Y):
        X_tensor = X.tensor() if hasattr(X, 'tensor') else X
        Y_tensor = Y.tensor() if hasattr(Y, 'tensor') else Y
        
        # Get batch shapes
        X_batch_shape = X_tensor.shape[:-1]
        Y_batch_shape = Y_tensor.shape[:-1]
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, Y_batch_shape)
        
        # Expand tensors to broadcast shape
        X_expanded = X_tensor.expand(*out_batch_shape, 7)
        Y_expanded = Y_tensor.expand(*out_batch_shape, 7)
        
        # Call forward
        from .. import warpSE3_type
        out = SE3_Mul_fwd(pp.LieTensor(X_expanded, ltype=warpSE3_type), 
                         pp.LieTensor(Y_expanded, ltype=warpSE3_type))
        
        # Save for backward (only X is needed for Y_grad computation)
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
        
        # Compute gradients
        grad_X, grad_Y = SE3_Mul_bwd(X_expanded, grad_output)
        
        # Handle broadcasting reduction
        ndim_out = len(out_batch_shape)
        ndim_X = len(X_batch_shape)
        ndim_Y = len(Y_batch_shape)
        
        # Pad shapes to match output shape
        X_padded = (1,) * (ndim_out - ndim_X) + tuple(X_batch_shape)
        Y_padded = (1,) * (ndim_out - ndim_Y) + tuple(Y_batch_shape)
        
        # Find dimensions to reduce for X
        reduce_dims_X = []
        for i, (x_dim, out_dim) in enumerate(zip(X_padded, out_batch_shape)):
            if x_dim == 1 and out_dim > 1:
                reduce_dims_X.append(i)
        
        # Find dimensions to reduce for Y
        reduce_dims_Y = []
        for i, (y_dim, out_dim) in enumerate(zip(Y_padded, out_batch_shape)):
            if y_dim == 1 and out_dim > 1:
                reduce_dims_Y.append(i)
        
        # Reduce gradients if necessary
        if reduce_dims_X:
            grad_X = grad_X.sum(dim=reduce_dims_X, keepdim=True)
        grad_X = grad_X.view(*X_batch_shape, 7)
        
        if reduce_dims_Y:
            grad_Y = grad_Y.sum(dim=reduce_dims_Y, keepdim=True)
        grad_Y = grad_Y.view(*Y_batch_shape, 7)
        
        return grad_X, grad_Y

