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


def get_tolerances(dtype: torch.dtype) -> dict:
    """
    Get appropriate atol/rtol for torch.testing.assert_close based on dtype.
    
    Returns a dict with 'atol' and 'rtol' keys suitable for unpacking:
        torch.testing.assert_close(result, expected, **get_tolerances(dtype))
    """
    if dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float32:
        return {"atol": 1e-5, "rtol": 1e-5}
    elif dtype == torch.float64:
        return {"atol": 1e-10, "rtol": 1e-10}
    else:
        raise NotImplementedError(f"Unimplemented for {dtype=}")


@pytest.fixture(params=[torch.float32, torch.float64, torch.float16], ids=["fp32", "fp64", "fp16"])
def dtype(request):
    """Parametrize over supported dtypes."""
    return request.param


@pytest.fixture(params=["cuda", "cpu"], ids=["cuda", "cpu"])
def device(request):
    """Parametrize over supported devices."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device
