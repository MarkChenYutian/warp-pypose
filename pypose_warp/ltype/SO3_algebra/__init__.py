import torch
import pypose as pp
from pypose.lietensor.lietensor import so3Type

from .Exp import so3_Exp, so3_Exp_fwd, so3_Exp_bwd
from .Mat import so3_Mat, so3_Mat_fwd, so3_Mat_bwd
from .Jr import so3_Jr, so3_Jr_fwd, so3_Jr_bwd


class warp_so3Type(so3Type):    
    def Exp(self, x: pp.LieTensor) -> pp.LieTensor:
        """
        Compute the exponential map from so3 to SO3.
        
        Args:
            x: so3 LieTensor of shape (..., 3) - axis-angle representation
            
        Returns:
            SO3 LieTensor of shape (..., 4) - quaternion representation
        """
        return so3_Exp.apply(x)

    def matrix(self, input: pp.LieTensor) -> torch.Tensor:
        """
        Convert so3 (axis-angle) to 3x3 rotation matrix.
        
        This is more efficient than PyPose's default implementation which uses:
            X = input.Exp(); I = eye(3); return X.Act(I).transpose(-1,-2)
        
        Args:
            input: so3 LieTensor of shape (..., 3) - axis-angle representation
            
        Returns:
            Rotation matrix of shape (..., 3, 3)
        """
        return so3_Mat.apply(input)

    def Jr(self, input: pp.LieTensor) -> torch.Tensor:
        """
        Compute the right Jacobian Jr of so3.
        
        The right Jacobian relates infinitesimal perturbations to the exponential map:
            Exp(x + dx) ≈ Exp(x) @ Exp(Jr(x) @ dx)
        
        Formula:
            Jr = I - (1 - cos(θ)) / θ² * K + (θ - sin(θ)) / θ³ * K @ K
        
        where K = skew(x) and θ = ||x||.
        
        Args:
            input: so3 LieTensor of shape (..., 3) - axis-angle representation
            
        Returns:
            Right Jacobian tensor of shape (..., 3, 3)
        """
        return so3_Jr.apply(input)

    @classmethod
    def identity(cls, *size, **kwargs):
        from ... import warpso3_type
        data = torch.tensor([0., 0., 0.], **kwargs)
        return pp.LieTensor(data.repeat(size+(1,)), ltype=warpso3_type)


warpso3_type = warp_so3Type()
