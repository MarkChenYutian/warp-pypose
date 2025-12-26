import torch
import pypose as pp
from .fwd import SO3_Jinvp_fwd
from .bwd import SO3_Jinvp_bwd

__all__ = ['SO3_Jinvp', 'SO3_Jinvp_fwd', 'SO3_Jinvp_bwd']


class SO3_Jinvp(torch.autograd.Function):
    """
    Compute Jinvp: Jl_inv(Log(X)) @ p
    
    Maps a tangent vector p through the inverse left Jacobian of the Log map.
    This is the adjoint-corrected transport operation for optimization on SO3.
    
    Forward:
        out = Jl_inv(Log(X)) @ p
        
    Args:
        X: SO3 LieTensor of shape (..., 4) - quaternion representation
        p: Tensor of shape (..., 3) - so3 tangent vector
        
    Returns:
        so3 LieTensor of shape (broadcast(...), 3) - transformed tangent vector
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X: pp.LieTensor, p: torch.Tensor) -> pp.LieTensor:
        out = SO3_Jinvp_fwd(X, p)
        
        # Save original batch shapes for gradient reduction in backward
        X_batch_shape = X.shape[:-1]
        p_batch_shape = p.shape[:-1]
        out_batch_shape = out.shape[:-1]
        
        # Expand X and p to broadcast shape for backward computation
        X_expanded = X.tensor().expand(*out_batch_shape, 4)
        p_expanded = p.expand(*out_batch_shape, 3)
        
        ctx.save_for_backward(X_expanded, p_expanded)
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
        
        # grad_output may be a LieTensor, extract tensor if needed
        if hasattr(grad_output, 'tensor'):
            grad_output = grad_output.tensor()
        
        # Compute gradients at broadcast shape
        grad_X, grad_p = SO3_Jinvp_bwd(X_expanded, p_expanded, grad_output)
        
        # Reduce gradients to original shapes by summing over broadcast dims
        ndim_out = len(out_batch_shape)
        ndim_X = len(X_batch_shape)
        ndim_p = len(p_batch_shape)
        
        # Pad shapes to same length for comparison
        X_padded = (1,) * (ndim_out - ndim_X) + tuple(X_batch_shape)
        p_padded = (1,) * (ndim_out - ndim_p) + tuple(p_batch_shape)
        
        # Find dims to reduce for X
        reduce_dims_X = []
        for i, (x_dim, out_dim) in enumerate(zip(X_padded, out_batch_shape)):
            if x_dim == 1 and out_dim > 1:
                reduce_dims_X.append(i)
        
        # Find dims to reduce for p
        reduce_dims_p = []
        for i, (p_dim, out_dim) in enumerate(zip(p_padded, out_batch_shape)):
            if p_dim == 1 and out_dim > 1:
                reduce_dims_p.append(i)
        
        # Reduce and reshape
        if reduce_dims_X:
            grad_X = grad_X.sum(dim=reduce_dims_X, keepdim=True)
        grad_X = grad_X.view(*X_batch_shape, 4)
        
        if reduce_dims_p:
            grad_p = grad_p.sum(dim=reduce_dims_p, keepdim=True)
        grad_p = grad_p.view(*p_batch_shape, 3)
        
        return grad_X, grad_p

