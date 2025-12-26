import torch
import pypose as pp
from .fwd import SE3_Act_fwd
from .bwd import SE3_Act_bwd


class SE3_Act(torch.autograd.Function):
    """
    SE3 Action on 3D points.
    
    Forward: out = t + R @ p
    where X = (t, q) is SE3 with translation t and rotation quaternion q.
    
    Backward follows PyPose's analytical gradients:
        X_grad[:3] = grad_output (translation gradient)
        X_grad[3:6] = cross(out, grad_output) (rotation Lie algebra gradient)
        X_grad[6] = 0 (w component always zero)
        p_grad = R^T @ grad_output
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
        out = SE3_Act_fwd(X, p)
        
        # Save original batch shapes for gradient reduction in backward
        X_batch_shape = X.shape[:-1]
        p_batch_shape = p.shape[:-1]
        out_batch_shape = out.shape[:-1]
        
        # Expand X to broadcast shape for backward computation
        X_expanded = X.tensor().expand(*out_batch_shape, 7)
        ctx.save_for_backward(X_expanded, out)
        ctx.X_batch_shape = X_batch_shape
        ctx.p_batch_shape = p_batch_shape
        ctx.out_batch_shape = out_batch_shape
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X_expanded, out = ctx.saved_tensors
        X_batch_shape = ctx.X_batch_shape
        p_batch_shape = ctx.p_batch_shape
        out_batch_shape = ctx.out_batch_shape
        
        # Compute gradients at broadcast shape
        grad_X, grad_p = SE3_Act_bwd(X_expanded, out, grad_output)
        
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
        grad_X = grad_X.view(*X_batch_shape, 7)
        
        if reduce_dims_p:
            grad_p = grad_p.sum(dim=reduce_dims_p, keepdim=True)
        grad_p = grad_p.view(*p_batch_shape, 3)
        
        return grad_X, grad_p

