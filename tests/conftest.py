# Initialize the paths
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Initialize the warp context
import warp as wp
wp.init()


# Fixtures and utilities
import pytest
import torch
from enum import Enum


# =============================================================================
# Operator Enum for Tolerance Registry
# =============================================================================

class Operator(Enum):
    """Enum of all operators for tolerance lookups."""
    # SO3 Group operators
    SO3_Mat = "SO3_Mat"
    SO3_Log = "SO3_Log"
    SO3_Mul = "SO3_Mul"
    SO3_Act = "SO3_Act"
    SO3_Act4 = "SO3_Act4"
    SO3_AdjXa = "SO3_AdjXa"
    SO3_AdjTXa = "SO3_AdjTXa"
    SO3_Jinvp = "SO3_Jinvp"
    # SO3 Algebra operators
    so3_Exp = "so3_Exp"
    so3_Mat = "so3_Mat"
    so3_Jr = "so3_Jr"
    # SE3 Group operators
    SE3_Act = "SE3_Act"
    SE3_Act4 = "SE3_Act4"
    SE3_Inv = "SE3_Inv"
    SE3_Mul = "SE3_Mul"
    SE3_AdjXa = "SE3_AdjXa"
    SE3_AdjTXa = "SE3_AdjTXa"
    SE3_Jinvp = "SE3_Jinvp"


# =============================================================================
# Tolerance Registry
# =============================================================================

# Default forward tolerances
_FWD_DEFAULTS = {
    torch.float16: {"atol": 2e-2, "rtol": 2e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float64: {"atol": 1e-10, "rtol": 1e-10},
}

# Default backward tolerances (looser due to different computational paths)
_BWD_DEFAULTS = {
    torch.float32: {"atol": 5e-3, "rtol": 5e-3},
    torch.float64: {"atol": 1e-10, "rtol": 1e-10},
}

# Operator-specific forward overrides
_FWD_OVERRIDES: dict[Operator, dict[torch.dtype, dict]] = {
    Operator.so3_Jr: {torch.float32: {"atol": 1e-4, "rtol": 1e-4}},
    # SE3_Jinvp involves multiple complex operations (Log, Jl_inv, calcQ) accumulating error
    Operator.SE3_Jinvp: {
        torch.float32: {"atol": 5e-4, "rtol": 5e-4},
        torch.float64: {"atol": 1e-9, "rtol": 1e-9},
    },
}

# Operator-specific backward overrides
_BWD_OVERRIDES: dict[Operator, dict[torch.dtype, dict]] = {
    Operator.SO3_Mat: {torch.float32: {"atol": 1e-4, "rtol": 1e-4}},
    Operator.so3_Mat: {torch.float32: {"atol": 1e-4, "rtol": 1e-4}},
    Operator.so3_Jr: {torch.float32: {"atol": 1e-4, "rtol": 1e-4}},
}


def get_fwd_tolerances(dtype: torch.dtype, operator: Operator = None) -> dict:
    """
    Get forward pass tolerances with optional operator-specific overrides.
    
    Args:
        dtype: The tensor dtype (float16, float32, float64)
        operator: Optional operator for specific overrides
        
    Returns:
        Dict with 'atol' and 'rtol' keys for torch.testing.assert_close
    """
    if operator and operator in _FWD_OVERRIDES and dtype in _FWD_OVERRIDES[operator]:
        return _FWD_OVERRIDES[operator][dtype]
    return _FWD_DEFAULTS[dtype]


def get_bwd_tolerances(dtype: torch.dtype, operator: Operator = None) -> dict:
    """
    Get backward pass tolerances with optional operator-specific overrides.
    
    Backward tolerances are looser than forward because analytical gradients
    and PyTorch autograd compute the same mathematical derivative through
    different computational paths, accumulating different rounding errors.
    
    Args:
        dtype: The tensor dtype (float32, float64) - fp16 not supported for backward
        operator: Optional operator for specific overrides
        
    Returns:
        Dict with 'atol' and 'rtol' keys for torch.testing.assert_close
    """
    if operator and operator in _BWD_OVERRIDES and dtype in _BWD_OVERRIDES[operator]:
        return _BWD_OVERRIDES[operator][dtype]
    return _BWD_DEFAULTS[dtype]


# Legacy alias for backward compatibility
def get_tolerances(dtype: torch.dtype) -> dict:
    """Legacy alias for get_fwd_tolerances. Prefer get_fwd_tolerances for new code."""
    return get_fwd_tolerances(dtype)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(params=[torch.float32, torch.float64, torch.float16], ids=["fp32", "fp64", "fp16"])
def dtype(request):
    """All dtypes for forward tests."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def dtype_bwd(request):
    """Exclude fp16 for backward tests (numerically unstable in gradient chains)."""
    return request.param


@pytest.fixture(params=["cuda", "cpu"], ids=["cuda", "cpu"])
def device(request):
    """Parametrize over supported devices."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device
