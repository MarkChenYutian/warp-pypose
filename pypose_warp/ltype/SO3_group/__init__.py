import torch
import pypose as pp
from pypose.lietensor.utils import LieType

from .Act   import SO3_Act, SO3_Act_fwd, SO3_Act_bwd
from .Log   import SO3_Log, SO3_Log_fwd, SO3_Log_bwd
from .Inv   import SO3_Inv, SO3_Inv_fwd, SO3_Inv_bwd
from .Mul   import SO3_Mul, SO3_Mul_fwd, SO3_Mul_bwd
from .AdjXa import SO3_AdjXa, SO3_AdjXa_fwd, SO3_AdjXa_bwd


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
        return SO3_Log.apply(X)
    
    def Inv(self, X: pp.LieTensor) -> pp.LieTensor:
        return SO3_Inv.apply(X)
    
    def Mul(self, X: pp.LieTensor, Y):
        # Transform on transform (LieTensor @ LieTensor)
        if not self.on_manifold and isinstance(Y, pp.LieTensor) and not Y.ltype.on_manifold:
            return SO3_Mul.apply(X, Y)
        # Transform on points (LieTensor @ Tensor)
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return pp.LieTensor(torch.mul(X.tensor(), Y), ltype=pp.SO3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def Adj(self, X: pp.LieTensor, a: pp.LieTensor) -> pp.LieTensor:
        return SO3_AdjXa.apply(X, a)

warpSO3_type = warp_SO3Type()
