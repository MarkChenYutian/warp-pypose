"""Benchmarking for SE3_Jinvp."""
import torch
from torch.utils.benchmark import Timer
import pypose as pp
from pypose_warp.ltype.SE3_group import SE3_Jinvp, SE3_Jinvp_fwd
from pypose_warp import to_warp_backend


def bench_forward(batch_size: int, device: str, dtype: torch.dtype) -> tuple:
    """
    Benchmark forward pass of SE3_Jinvp.
    
    Returns:
        (pypose_timer, warp_timer): Timer objects for PyPose and Warp implementations.
    """
    X = pp.randn_SE3(batch_size, device=device, dtype=dtype)
    p = pp.randn_se3(batch_size, device=device, dtype=dtype)
    X_warp = to_warp_backend(X)

    # PyPose reference
    pypose_timer = Timer(
        stmt="X.Jinvp(p)",
        globals={"X": X, "p": p},
        label="SE3_Jinvp forward",
        sub_label=f"{device}-{dtype}".replace("torch.", ""),
        description="PyPose",
    )

    # Warp implementation
    warp_timer = Timer(
        stmt="SE3_Jinvp_fwd(X, p)",
        globals={"SE3_Jinvp_fwd": SE3_Jinvp_fwd, "X": X_warp, "p": p},
        label="SE3_Jinvp forward",
        sub_label=f"{device}-{dtype}".replace("torch.", ""),
        description="Warp",
    )

    return pypose_timer.blocked_autorange(), warp_timer.blocked_autorange()


def bench_backward(batch_size: int, device: str, dtype: torch.dtype) -> tuple:
    """
    Benchmark backward pass of SE3_Jinvp.
    
    Returns:
        (pypose_timer, warp_timer): Timer objects for PyPose and Warp implementations.
    """
    X_base = pp.randn_SE3(batch_size, device=device, dtype=dtype)
    p_base = pp.randn_se3(batch_size, device=device, dtype=dtype)

    # PyPose reference
    def pypose_backward():
        X = X_base.clone().requires_grad_(True)
        p = p_base.clone().requires_grad_(True)
        out = X.Jinvp(p)
        out.sum().backward()
        return X.grad, p.grad

    pypose_timer = Timer(
        stmt="benchmark_fn()",
        globals={"benchmark_fn": pypose_backward},
        label="SE3_Jinvp backward",
        sub_label=f"{device}-{dtype}".replace("torch.", ""),
        description="PyPose",
    )

    # Warp implementation
    X_warp_base = to_warp_backend(X_base)

    def warp_backward():
        X = X_warp_base.clone().requires_grad_(True)
        p = p_base.clone().requires_grad_(True)
        out = SE3_Jinvp.apply(X, p)
        out.sum().backward()
        return X.grad, p.grad

    warp_timer = Timer(
        stmt="benchmark_fn()",
        globals={"benchmark_fn": warp_backward, "SE3_Jinvp": SE3_Jinvp},
        label="SE3_Jinvp backward",
        sub_label=f"{device}-{dtype}".replace("torch.", ""),
        description="Warp",
    )

    return pypose_timer.blocked_autorange(), warp_timer.blocked_autorange()

