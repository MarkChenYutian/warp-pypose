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
    SO3_AddExp = "SO3_AddExp"
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
    SE3_Mat = "SE3_Mat"


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
        torch.float16: {"atol": 3e-2, "rtol": 3e-2},  # Looser for FP16 due to accumulated errors
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
# FP16 Numerical Stability Utilities
# =============================================================================

import pypose as pp


def skip_if_nan_inputs(*tensors):
    """
    Skip test if any input tensor contains NaN values.
    
    This is needed because PyPose's randn_SE3 can produce NaN values
    in FP16 due to upstream numerical issues (~0.07% of elements).
    
    Usage:
        skip_if_nan_inputs(X.tensor(), p.tensor())
    """
    for t in tensors:
        tensor = t.tensor() if hasattr(t, 'tensor') else t
        if torch.isnan(tensor).any():
            pytest.skip("Input data contains NaN (PyPose fp16 randn issue)")


def _get_ltype_class(lietensor):
    """Get the PyPose LieTensor class for a given lietensor."""
    ltype = lietensor.ltype
    if ltype == pp.SO3_type:
        return pp.SO3
    elif ltype == pp.SE3_type:
        return pp.SE3
    elif ltype == pp.so3_type:
        return pp.so3
    elif ltype == pp.se3_type:
        return pp.se3
    else:
        # Fallback - try to use the type directly
        return type(lietensor)


def compute_reference_fp32(lietensor, method_name, *args):
    """
    Compute reference using FP32 precision, cast result back to original dtype.
    
    This avoids numerical instability in PyPose's FP16 computations while
    still validating our more numerically stable FP16 Warp implementation.
    
    Args:
        lietensor: The LieTensor to call the method on (e.g., SE3, SO3)
        method_name: Name of the method to call (e.g., 'Jinvp', 'Act')
        *args: Additional arguments to pass to the method
        
    Returns:
        Result tensor in the original dtype
        
    Example:
        expected = compute_reference_fp32(X, 'Jinvp', p)
        # Equivalent to: X_fp32.Jinvp(p_fp32).tensor().to(original_dtype)
    """
    original_dtype = lietensor.tensor().dtype
    
    if original_dtype == torch.float16:
        # Upcast LieTensor to FP32 using the correct PyPose class
        ltype_cls = _get_ltype_class(lietensor)
        lietensor_fp32 = ltype_cls(lietensor.tensor().float())
        
        # Upcast arguments to FP32
        args_fp32 = []
        for arg in args:
            if hasattr(arg, 'tensor') and hasattr(arg, 'ltype'):
                # It's a LieTensor
                arg_cls = _get_ltype_class(arg)
                args_fp32.append(arg_cls(arg.tensor().float()))
            elif isinstance(arg, torch.Tensor):
                args_fp32.append(arg.float())
            else:
                args_fp32.append(arg)
        
        # Compute in FP32
        method = getattr(lietensor_fp32, method_name)
        result_fp32 = method(*args_fp32)
        
        # Downcast result to FP16
        if hasattr(result_fp32, 'tensor'):
            return result_fp32.tensor().half()
        elif isinstance(result_fp32, torch.Tensor):
            return result_fp32.half()
        else:
            return result_fp32
    else:
        # For FP32/FP64, compute directly
        method = getattr(lietensor, method_name)
        result = method(*args)
        return result.tensor() if hasattr(result, 'tensor') else result


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
