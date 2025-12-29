import torch
import pypose as pp
from pypose.lietensor.lietensor import se3Type

from .Exp import se3_Exp, se3_Exp_fwd, se3_Exp_bwd
from .Mat import se3_Mat, se3_Mat_fwd, se3_Mat_bwd


class warp_se3Type(se3Type):
    def Exp(self, x: pp.LieTensor) -> pp.LieTensor:
        """
        Compute the exponential map from se3 to SE3.
        
        Args:
            x: se3 LieTensor of shape (..., 6) - [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z]
            
        Returns:
            SE3 LieTensor of shape (..., 7) - [tx, ty, tz, qx, qy, qz, qw]
        """
        return se3_Exp.apply(x)

    def matrix(self, input: pp.LieTensor) -> torch.Tensor:
        """
        Convert se3 twist to 4x4 transformation matrix.
        
        This is equivalent to:
            input.Exp().matrix()
        
        But computed in a single fused kernel for efficiency.
        
        Args:
            input: se3 LieTensor of shape (..., 6) - [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z]
            
        Returns:
            Transformation matrix of shape (..., 4, 4)
        """
        return se3_Mat.apply(input)

    @classmethod
    def identity(cls, *size, **kwargs):
        from . import warpse3_type
        data = torch.tensor([0., 0., 0., 0., 0., 0.], **kwargs)
        return pp.LieTensor(data.repeat(size+(1,)), ltype=warpse3_type)


warpse3_type = warp_se3Type()
