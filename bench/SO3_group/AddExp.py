"""
Benchmark for SO3 AddExp: fused Exp(delta) * X operation.

Compares performance of:
- Warp backend: fused kernel
- PyPose backend: separate Exp + Mul operations
"""
import argparse
import torch
import typing as T
import pypose as pp
import warp as wp

wp.init()

from pypose_warp import to_warp_backend
from pypose_warp.ltype.SO3_group import SO3_AddExp
from torch.utils.benchmark import Timer


def bench_forward(num: int, device: T.Literal["cpu", "cuda"], dtype: torch.dtype, kind: T.Literal["n-n", "1-n", "n-1"] = "n-n"):
    """
    Benchmark forward pass of AddExp.
    
    Note: kind parameter is accepted for API consistency but AddExp doesn't support
    broadcasting patterns - both delta and X must have the same batch shape.
    """
    X = pp.randn_SO3(num, device=device, dtype=dtype)
    delta = torch.randn(num, 3, device=device, dtype=dtype) * 0.1
    
    X_warp = to_warp_backend(X)
    
    # PyPose: separate Exp + Mul
    def pp_addexp():
        delta_exp = pp.so3(delta).Exp()
        return delta_exp @ X
    
    # Warp: fused kernel
    def wp_addexp():
        return SO3_AddExp.apply(delta, X_warp)
    
    pp_timer = Timer(stmt="pp_addexp()", globals=dict(pp_addexp=pp_addexp))
    wp_timer = Timer(stmt="wp_addexp()", globals=dict(wp_addexp=wp_addexp))
    
    pp_bench = pp_timer.adaptive_autorange()
    wp_bench = wp_timer.adaptive_autorange()
    return pp_bench, wp_bench


def bench_backward(num: int, device: T.Literal["cpu", "cuda"], dtype: torch.dtype, kind: T.Literal["n-n", "1-n", "n-1"] = "n-n"):
    """
    Benchmark backward pass of AddExp.
    
    Note: kind parameter is accepted for API consistency but AddExp doesn't support
    broadcasting patterns - both delta and X must have the same batch shape.
    """
    # PyPose backward
    def pp_backward():
        X = pp.randn_SO3(num, device=device, dtype=dtype, requires_grad=True)
        delta = torch.randn(num, 3, device=device, dtype=dtype, requires_grad=True) * 0.1
        delta_exp = pp.so3(delta).Exp()
        result = delta_exp @ X
        result.sum().backward()
    
    # Warp backward
    def wp_backward():
        X = pp.randn_SO3(num, device=device, dtype=dtype, requires_grad=True)
        delta = torch.randn(num, 3, device=device, dtype=dtype, requires_grad=True) * 0.1
        X_warp = to_warp_backend(X)
        result = SO3_AddExp.apply(delta, X_warp)
        result.sum().backward()
    
    pp_timer = Timer(stmt="pp_backward()", globals=dict(pp_backward=pp_backward))
    wp_timer = Timer(stmt="wp_backward()", globals=dict(wp_backward=wp_backward))
    
    pp_bench = pp_timer.adaptive_autorange()
    wp_bench = wp_timer.adaptive_autorange()
    return pp_bench, wp_bench


DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark SO3 AddExp forward/backward")
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="fwd", help="Benchmark forward or backward pass")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to run on")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "fp64"], default="fp32", help="Data type")
    parser.add_argument("--size", type=int, default=10000, help="Batch size")
    args = parser.parse_args()
    
    dtype = DTYPE_MAP[args.dtype]
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Benchmarking SO3.AddExp {args.mode} | device={args.device} | dtype={args.dtype} | size={args.size}")
    print("-" * 80)
    
    if args.mode == "fwd":
        pp_bench, wp_bench = bench_forward(args.size, args.device, dtype)
    else:
        pp_bench, wp_bench = bench_backward(args.size, args.device, dtype)
    
    print(f"PyPose (Exp+Mul): {pp_bench}")
    print(f"Warp (fused):     {wp_bench}")
    print("-" * 80)
    
    speedup = pp_bench.median / wp_bench.median
    print(f"Speedup: {speedup:.2f}x {'(Warp faster)' if speedup > 1 else '(PyPose faster)'}")


if __name__ == "__main__":
    main()

