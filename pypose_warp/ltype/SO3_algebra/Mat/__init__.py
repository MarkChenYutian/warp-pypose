import torch
import pypose as pp
from .fwd import so3_Mat_fwd


class so3_Mat(torch.autograd.Function):
    """
    Convert so3 (axis-angle) to 3x3 rotation matrix.
    
    This is equivalent to PyPose's so3Type.matrix() method but more efficient
    as it computes directly in a single warp kernel for the forward pass.
    
    Forward: x (so3) -> R (rotation matrix)
    Uses: Exp(x) -> quat -> quat_to_matrix
    
    Backward: Uses PyTorch autograd by composing Exp and quat_to_matrix
    to get numerically correct gradients.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(ctx, x):
        # Convert LieTensor to tensor if needed
        x_tensor = x.tensor() if hasattr(x, 'tensor') else x
        
        # Use optimized warp forward
        out = so3_Mat_fwd(x)
        
        # Save input tensor for backward (we'll recompute through PyTorch)
        ctx.save_for_backward(x_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_tensor, = ctx.saved_tensors
        
        # Recompute forward through PyTorch autograd for correct gradients
        # This gives us correct chain rule through Exp and quat_to_matrix
        # We need enable_grad() since backward runs in no_grad context
        with torch.enable_grad():
            x_with_grad = x_tensor.detach().requires_grad_(True)
            so3 = pp.LieTensor(x_with_grad, ltype=pp.so3_type)
            R = so3.matrix()  # Uses PyPose's Exp -> Act -> transpose
            
            # Compute gradient using torch.autograd.grad
            grad_x, = torch.autograd.grad(R, x_with_grad, grad_output)
        
        return grad_x


# Re-export for compatibility (bwd is not used directly anymore)
def so3_Mat_bwd(x, grad_output):
    """Backward pass using PyTorch autograd (for standalone use)."""
    x_with_grad = x.detach().requires_grad_(True)
    so3 = pp.LieTensor(x_with_grad, ltype=pp.so3_type)
    R = so3.matrix()
    R.backward(grad_output)
    return x_with_grad.grad
