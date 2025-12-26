import torch
import pypose as pp
from pypose.lietensor.utils import LieType

from .Exp import so3_Exp, so3_Exp_fwd, so3_Exp_bwd


class warp_so3Type(LieType):
    def __init__(self):
        super().__init__(dimension=3, embedding=4, manifold=3)
    
    def Exp(self, x: pp.LieTensor) -> pp.LieTensor:
        """
        Compute the exponential map from so3 to SO3.
        
        Args:
            x: so3 LieTensor of shape (..., 3) - axis-angle representation
            
        Returns:
            SO3 LieTensor of shape (..., 4) - quaternion representation
        """
        return so3_Exp.apply(x)
