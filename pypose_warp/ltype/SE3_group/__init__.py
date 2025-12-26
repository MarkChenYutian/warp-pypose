import torch
import pypose as pp
from pypose.lietensor.lietensor import SE3Type

from .Act import SE3_Act, SE3_Act_fwd, SE3_Act_bwd
from .Act4 import SE3_Act4, SE3_Act4_fwd, SE3_Act4_bwd


class warp_SE3Type(SE3Type):
    def Act(self, X: pp.LieTensor, p: torch.Tensor):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1] == 3 or p.shape[-1] == 4, "Invalid Tensor Dimension"
        out: torch.Tensor
        if p.shape[-1] == 3:
            out = SE3_Act.apply(X, p)
        else:
            out = SE3_Act4.apply(X, p)
        return out


warpSE3_type = warp_SE3Type()
