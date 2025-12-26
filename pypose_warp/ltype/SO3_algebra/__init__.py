import torch
import pypose as pp
from pypose.lietensor.lietensor import so3Type

from .Exp import so3_Exp, so3_Exp_fwd, so3_Exp_bwd
from .Mat import so3_Mat, so3_Mat_fwd, so3_Mat_bwd


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


warpso3_type = warp_so3Type()
