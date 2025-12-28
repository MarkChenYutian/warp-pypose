import torch
import pypose as pp

from .fwd import SE3_Jinvp_fwd
from .bwd import SE3_Jinvp_bwd


class SE3_Jinvp(torch.autograd.Function):
    """
    SE3 Jinvp: Jl_inv(Log(X)) @ p
    
    Maps a tangent vector p through the inverse left Jacobian of the Log map.
    
    Forward: out = se3_Jl_inv(SE3_Log(X)) @ p
    
    Backward:
        grad_p = se3_Jl_inv^T @ grad_out
        grad_X = SE3_Log_backward(grad_log_X)
    """
    
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X, p):
        X_tensor = X.tensor() if hasattr(X, 'tensor') else X
        p_tensor = p.tensor() if hasattr(p, 'tensor') else p
        
        # Get batch shapes
        X_batch_shape = X_tensor.shape[:-1]
        p_batch_shape = p_tensor.shape[:-1]
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
        
        # Expand tensors to broadcast shape
        X_expanded = X_tensor.expand(*out_batch_shape, 7)
        p_expanded = p_tensor.expand(*out_batch_shape, 6)
        
        # Call forward
        from .. import warpSE3_type
        out = SE3_Jinvp_fwd(
            pp.LieTensor(X_expanded, ltype=warpSE3_type),
            pp.LieTensor(p_expanded, ltype=pp.se3_type)
        )
        
        # Save for backward
        ctx.save_for_backward(X_expanded.contiguous(), p_expanded.contiguous())
        ctx.X_batch_shape = X_batch_shape
        ctx.p_batch_shape = p_batch_shape
        ctx.out_batch_shape = out_batch_shape
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X_expanded, p_expanded = ctx.saved_tensors
        X_batch_shape = ctx.X_batch_shape
        p_batch_shape = ctx.p_batch_shape
        out_batch_shape = ctx.out_batch_shape
        
        # Compute gradients
        grad_X, grad_p = SE3_Jinvp_bwd(X_expanded, p_expanded, grad_output)
        
        # Handle broadcasting reduction
        ndim_out = len(out_batch_shape)
        ndim_X = len(X_batch_shape)
        ndim_p = len(p_batch_shape)
        
        # Pad shapes to match output shape
        X_padded = (1,) * (ndim_out - ndim_X) + tuple(X_batch_shape)
        p_padded = (1,) * (ndim_out - ndim_p) + tuple(p_batch_shape)
        
        # Find dimensions to reduce for X
        reduce_dims_X = []
        for i, (x_dim, out_dim) in enumerate(zip(X_padded, out_batch_shape)):
            if x_dim == 1 and out_dim > 1:
                reduce_dims_X.append(i)
        
        # Find dimensions to reduce for p
        reduce_dims_p = []
        for i, (p_dim, out_dim) in enumerate(zip(p_padded, out_batch_shape)):
            if p_dim == 1 and out_dim > 1:
                reduce_dims_p.append(i)
        
        # Reduce gradients if necessary
        if reduce_dims_X:
            grad_X = grad_X.sum(dim=reduce_dims_X, keepdim=True)
        grad_X = grad_X.view(*X_batch_shape, 7)
        
        if reduce_dims_p:
            grad_p = grad_p.sum(dim=reduce_dims_p, keepdim=True)
        grad_p = grad_p.view(*p_batch_shape, 6)
        
        return grad_X, grad_p

