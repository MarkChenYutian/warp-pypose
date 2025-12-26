import torch
import pypose as pp

from .fwd import SO3_Act4_fwd
from .bwd import SO3_Act4_bwd


class SO3_Act4(torch.autograd.Function):
    """
    Autograd wrapper for SO3_Act4 forward and backward.
    
    SO3_Act4 applies a rotation to 4D homogeneous points:
      out = [R @ p[:3], p[3]]
    where R is the rotation matrix corresponding to quaternion X.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(X: pp.LieTensor, p: torch.Tensor) -> torch.Tensor:
        X_tensor = X.tensor()
        
        # Get original batch shapes for backward gradient reduction
        X_batch_shape = X_tensor.shape[:-1]
        p_batch_shape = p.shape[:-1]
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
        
        # Expand X and p to broadcast shape
        X_expanded = X_tensor.expand(*out_batch_shape, 4)
        p_expanded = p.expand(*out_batch_shape, 4)
        
        # Run forward pass
        out = SO3_Act4_fwd(pp.LieTensor(X_expanded, ltype=pp.SO3_type), p_expanded)
        
        # Save expanded tensors and original shapes for backward
        # We need to save X_expanded because backward needs it at the output shape
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, p = inputs
        X_tensor = X.tensor()
        
        X_batch_shape = X_tensor.shape[:-1]
        p_batch_shape = p.shape[:-1]
        out_batch_shape = torch.broadcast_shapes(X_batch_shape, p_batch_shape)
        
        X_expanded = X_tensor.expand(*out_batch_shape, 4)
        
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
        
        # Compute gradients at the output (broadcasted) shape
        grad_X, grad_p = SO3_Act4_bwd(X_expanded, out, grad_output)
        
        # Reduce gradients to original input shapes by summing over broadcasted dims
        # For X:
        if X_batch_shape != out_batch_shape:
            # Find dimensions that were broadcasted (where X had size 1 or was missing)
            X_batch_ndim = len(X_batch_shape)
            out_batch_ndim = len(out_batch_shape)
            
            # Prepend 1s to X_batch_shape to match out_batch_ndim
            X_batch_padded = (1,) * (out_batch_ndim - X_batch_ndim) + X_batch_shape
            
            # Sum over broadcasted dimensions
            reduce_dims = []
            for i, (x_size, o_size) in enumerate(zip(X_batch_padded, out_batch_shape)):
                if x_size == 1 and o_size != 1:
                    reduce_dims.append(i)
            
            if reduce_dims:
                grad_X = grad_X.sum(dim=reduce_dims, keepdim=True)
            
            # Reshape back to original X_batch_shape
            grad_X = grad_X.view(*X_batch_shape, 4)
        
        # For p:
        if p_batch_shape != out_batch_shape:
            p_batch_ndim = len(p_batch_shape)
            out_batch_ndim = len(out_batch_shape)
            
            p_batch_padded = (1,) * (out_batch_ndim - p_batch_ndim) + p_batch_shape
            
            reduce_dims = []
            for i, (p_size, o_size) in enumerate(zip(p_batch_padded, out_batch_shape)):
                if p_size == 1 and o_size != 1:
                    reduce_dims.append(i)
            
            if reduce_dims:
                grad_p = grad_p.sum(dim=reduce_dims, keepdim=True)
            
            grad_p = grad_p.view(*p_batch_shape, 4)
        
        return grad_X, grad_p

