import argparse
import torch
import typing as T
import pypose as pp
import warp as wp

wp.init()

from pypose_warp import to_warp_backend
from pypose_warp.ltype.SE3_group import SE3_AdjXa
from torch.utils.benchmark import Timer


def bench_forward(num: int, device: T.Literal["cpu", "cuda"], dtype: torch.dtype):
    """
    Benchmark forward pass of SE3 AdjXa.
    
    Args:
        num: Batch size
        device: Device to run on
        dtype: Data type
    """
    pp_X = pp.randn_SE3(num, device=device, dtype=dtype)
    pp_a = pp.randn_se3(num, device=device, dtype=dtype)
    wp_X = to_warp_backend(pp_X)
    
    pp_timer = Timer(stmt="X.Adj(a)", globals=dict(X=pp_X, a=pp_a))
    wp_timer = Timer(stmt="X.Adj(a)", globals=dict(X=wp_X, a=pp_a))
    
    pp_bench = pp_timer.adaptive_autorange()
    wp_bench = wp_timer.adaptive_autorange()
    return pp_bench, wp_bench


def bench_backward(num: int, device: T.Literal["cpu", "cuda"], dtype: torch.dtype):
    """
    Benchmark backward pass of SE3 AdjXa.
    
    Args:
        num: Batch size
        device: Device to run on
        dtype: Data type
    """
    # PyPose backward
    def pp_backward():
        X = pp.randn_SE3(num, device=device, dtype=dtype, requires_grad=True)
        a = pp.randn_se3(num, device=device, dtype=dtype, requires_grad=True)
        result = X.Adj(a)
        result.sum().backward()
    
    # Warp backward (using SE3_AdjXa.apply directly)
    def wp_backward():
        X = pp.randn_SE3(num, device=device, dtype=dtype, requires_grad=True)
        a = pp.randn_se3(num, device=device, dtype=dtype, requires_grad=True)
        result = SE3_AdjXa.apply(X, a)
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
    parser = argparse.ArgumentParser(description="Benchmark SE3 AdjXa forward/backward")
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="fwd", help="Benchmark forward or backward pass")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to run on")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "fp64"], default="fp32", help="Data type")
    parser.add_argument("--size", type=int, default=10000, help="Batch size")
    args = parser.parse_args()
    
    dtype = DTYPE_MAP[args.dtype]
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Benchmarking SE3.Adj {args.mode} | device={args.device} | dtype={args.dtype} | size={args.size}")
    print("-" * 80)
    
    if args.mode == "fwd":
        pp_bench, wp_bench = bench_forward(args.size, args.device, dtype)
    else:
        pp_bench, wp_bench = bench_backward(args.size, args.device, dtype)
    
    print(f"PyPose:  {pp_bench}")
    print(f"Warp:    {wp_bench}")
    print("-" * 80)
    
    speedup = pp_bench.median / wp_bench.median
    print(f"Speedup: {speedup:.2f}x {'(Warp faster)' if speedup > 1 else '(PyPose faster)'}")


if __name__ == "__main__":
    main()

