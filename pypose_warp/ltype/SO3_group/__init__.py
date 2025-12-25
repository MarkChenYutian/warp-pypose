import torch
import pypose as pp
from pypose.lietensor.utils import LieType

from .Act import SO3_Act, SO3_Act_fwd, SO3_Act_bwd
from .Log import SO3_Log, SO3_Log_fwd


class warp_SO3Type(LieType):
    def __init__(self):
        super().__init__(dimension=4, embedding=4, manifold=3)
    
    def Act(self, X: pp.LieTensor, p: torch.Tensor):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        
        out: torch.Tensor
        if p.shape[-1]==3:
            out = SO3_Act.apply(X, p)
        else:
            raise NotImplementedError("SO3_Act4 not implemented yet!")
        return out
    
    def Log(self, X: pp.LieTensor) -> pp.LieTensor:
        assert not self.on_manifold
        return SO3_Log.apply(X)


warpSO3_type = warp_SO3Type()
