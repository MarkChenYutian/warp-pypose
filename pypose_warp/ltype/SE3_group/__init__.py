import torch
import pypose as pp
from torch import Tensor
from numbers import Number
from pypose.lietensor.lietensor import SE3Type, LieTensor
from pypose.lietensor.operation import broadcast_inputs

from .Act import SE3_Act, SE3_Act_fwd, SE3_Act_bwd
from .Act4 import SE3_Act4, SE3_Act4_fwd, SE3_Act4_bwd
from .Inv import SE3_Inv, SE3_Inv_fwd, SE3_Inv_bwd
from .Mul import SE3_Mul, SE3_Mul_fwd, SE3_Mul_bwd


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

    def Inv(self, X: pp.LieTensor) -> pp.LieTensor:
        out = SE3_Inv.apply(X)
        return LieTensor(out, ltype=self)

    def Mul(self, X: pp.LieTensor, Y: Number | Tensor | pp.LieTensor) -> Tensor | pp.LieTensor:
        """
        SE3 multiplication with dispatch logic following PyPose's SE3Type.Mul:
        
        1. Transform × Transform → SE3_Mul (compose two SE3 elements)
        2. Transform × Tensor → Act (apply transform to points)
        3. Scalar × Manifold → element-wise multiplication
        """
        X_tensor = X.tensor() if isinstance(X, LieTensor) else X
        
        # Case 1: Transform on transform (SE3 @ SE3)
        if not self.on_manifold and isinstance(Y, LieTensor) and not Y.ltype.on_manifold:
            Y_tensor = Y.tensor() if hasattr(Y, 'tensor') else Y
            input_tensors, out_shape = broadcast_inputs(X_tensor, Y_tensor)
            out = SE3_Mul.apply(
                LieTensor(input_tensors[0], ltype=self),
                LieTensor(input_tensors[1], ltype=self)
            )
            dim = -1 if out.nelement() != 0 else X_tensor.shape[-1]
            out = out.view(out_shape + (dim,))
            return LieTensor(out, ltype=self)
        
        # Case 2: Transform on points (SE3 @ Tensor)
        if not self.on_manifold and isinstance(Y, Tensor):
            return self.Act(X, Y)
        
        # Case 3: Scalar * manifold (element-wise multiplication)
        if self.on_manifold:
            return LieTensor(torch.mul(X_tensor, Y), ltype=self)
        
        raise NotImplementedError('Invalid __mul__ operation')


warpSE3_type = warp_SE3Type()
