import warp as wp
import pypose as pp
from pypose.lietensor.lietensor import LieType

from .ltype import warpSO3_type
wp.init()


_BACKEND_LIST: list[tuple[LieType, LieType | None]] = [
    # (Pypose LieType, Warp LieType)
    (pp.SO3_type  , warpSO3_type),
    (pp.SE3_type  , None),
    (pp.Sim3_type , None),
    (pp.RxSO3_type, None)
]
_PP_TO_WP = {pp_ltype : wp_ltype for pp_ltype, wp_ltype in _BACKEND_LIST}
_WP_TO_PP = {wp_ltype : pp_ltype for pp_ltype, wp_ltype in _BACKEND_LIST}


def to_warp_backend(x: pp.LieTensor) -> pp.LieTensor:
    """Swap the lietensor backend for accelerated compute"""
    if is_warp_backend(x): return x
    wp_ltype = _PP_TO_WP[x.ltype]
    
    if wp_ltype is None:
        raise NotImplementedError(f"Warp backend not implemented for pypose LieType {x.ltype}.")
    
    return pp.LieTensor(x.tensor(), ltype=wp_ltype)


def to_pypose_backend(x: pp.LieTensor) -> pp.LieTensor:
    """Swap the lietensor backend for better op coverage"""
    if is_pypose_backend(x): return x
    return pp.LieTensor(x.tensor(), ltype=_WP_TO_PP[x.ltype])


def is_warp_backend(x: pp.LieTensor) -> bool:
    return x.ltype in {
        warpSO3_type
    }


def is_pypose_backend(x: pp.LieTensor) -> bool:
    return x.ltype in {
        pp.SE3_type, pp.SO3_type, pp.RxSO3_type, pp.Sim3_type
    }
