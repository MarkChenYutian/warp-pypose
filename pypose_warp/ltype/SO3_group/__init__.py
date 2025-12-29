import torch
import pypose as pp
from pypose.lietensor.lietensor import SO3Type

from .Act    import SO3_Act, SO3_Act_fwd, SO3_Act_bwd
from .Act4   import SO3_Act4, SO3_Act4_fwd, SO3_Act4_bwd
from .Log    import SO3_Log, SO3_Log_fwd, SO3_Log_bwd
from .Mul    import SO3_Mul, SO3_Mul_fwd, SO3_Mul_bwd
from .AdjXa  import SO3_AdjXa, SO3_AdjXa_fwd, SO3_AdjXa_bwd
from .AdjTXa import SO3_AdjTXa, SO3_AdjTXa_fwd, SO3_AdjTXa_bwd
from .Jinvp  import SO3_Jinvp, SO3_Jinvp_fwd, SO3_Jinvp_bwd
from .Mat    import SO3_Mat, SO3_Mat_fwd, SO3_Mat_bwd
from .AddExp import SO3_AddExp, SO3_AddExp_fwd, SO3_AddExp_bwd


class warp_SO3Type(SO3Type):
    def Log(self, X: pp.LieTensor) -> pp.LieTensor:
        return SO3_Log.apply(X)
    
    def Act(self, X: pp.LieTensor, p: torch.Tensor):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        out: torch.Tensor
        if p.shape[-1]==3:
            out = SO3_Act.apply(X, p)
        else:
            out = SO3_Act4.apply(X, p)
        return out
    
    def Mul(self, X: pp.LieTensor, Y):
        # Transform on transform (LieTensor @ LieTensor)
        if not self.on_manifold and isinstance(Y, pp.LieTensor) and not Y.ltype.on_manifold:
            return SO3_Mul.apply(X, Y)
        # Transform on points (LieTensor @ Tensor)
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return pp.LieTensor(torch.mul(X.tensor(), Y), ltype=warpSO3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    # NOTE: we don't override the Inv since pypose's original Inv is faster than Warp
    #       (launching warp kernel incurs additional overhead)
    # def Inv(self, X: pp.LieTensor) -> pp.LieTensor:
    #     return SO3_Inv.apply(X)
    
    def Adj(self, X: pp.LieTensor, a: pp.LieTensor) -> pp.LieTensor:
        return SO3_AdjXa.apply(X, a)

    def AdjT(self, X: pp.LieTensor, a: pp.LieTensor) -> pp.LieTensor:
        return SO3_AdjTXa.apply(X, a)

    def Jinvp(self, X: pp.LieTensor, p: pp.LieTensor) -> pp.LieTensor:
        p_tensor = p.tensor() if isinstance(p, pp.LieTensor) else p
        return SO3_Jinvp.apply(X, p_tensor)

    def matrix(self, input: pp.LieTensor) -> torch.Tensor:
        """
        Convert SO3 quaternion to 3x3 rotation matrix.
        
        This is more efficient than PyPose's default implementation which uses:
            I = eye(3); return input.Act(I).transpose(-1,-2)
        
        Args:
            input: SO3 LieTensor of shape (..., 4) - quaternion representation
            
        Returns:
            Rotation matrix of shape (..., 3, 3)
        """
        return SO3_Mat.apply(input)

    @classmethod
    def add_(cls, input, other):
        """
        In-place update: input = Exp(other[..., :3]) * input.
        
        Uses fused AddExp kernel for better performance.
        
        Args:
            input: SO3 LieTensor to update in-place
            other: Tensor containing tangent space delta (uses first 3 components)
            
        Returns:
            input (modified in-place)
        """
        delta = other[..., :3]
        result = SO3_AddExp.apply(delta, input)
        return input.copy_(result)


warpSO3_type = warp_SO3Type()
