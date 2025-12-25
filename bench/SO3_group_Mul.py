import argparse
import torch
import typing as T
import pypose as pp
import warp as wp

wp.init()

from pypose_warp import to_warp_backend
from pypose_warp.ltype.SO3_group import SO3_Mul
from torch.utils.benchmark import Timer


def bench_forward(num: int, device: T.Literal["cpu", "cuda"], dtype: torch.dtype, kind: T.Literal["n-n", "1-n", "n-1"]):
    match kind:
        case "n-n": num_x, num_y = num, num
        case "1-n": num_x, num_y = 1  , num
        case "n-1": num_x, num_y = num, 1
    
    pp_x = pp.randn_SO3(num_x, device=device, dtype=dtype)
    pp_y = pp.randn_SO3(num_y, device=device, dtype=dtype)
    wp_x = to_warp_backend(pp_x)
    wp_y = to_warp_backend(pp_y)
    
    pp_timer = Timer(stmt="x @ y", globals=dict(x=pp_x, y=pp_y))
    wp_timer = Timer(stmt="x @ y", globals=dict(x=wp_x, y=wp_y))
    
    pp_bench = pp_timer.adaptive_autorange()
    wp_bench = wp_timer.adaptive_autorange()
    return pp_bench, wp_bench


def bench_backward(num: int, device: T.Literal["cpu", "cuda"], dtype: torch.dtype, kind: T.Literal["n-n", "1-n", "n-1"]):
    match kind:
        case "n-n": num_x, num_y = num, num
        case "1-n": num_x, num_y = 1  , num
        case "n-1": num_x, num_y = num, 1
    
    # PyPose backward
    def pp_backward():
        x = pp.randn_SO3(num_x, device=device, dtype=dtype, requires_grad=True)
        y = pp.randn_SO3(num_y, device=device, dtype=dtype, requires_grad=True)
        result = x @ y
        result.sum().backward()
    
    # Warp backward (using SO3_Mul.apply directly)
    def wp_backward():
        x = pp.randn_SO3(num_x, device=device, dtype=dtype, requires_grad=True)
        y = pp.randn_SO3(num_y, device=device, dtype=dtype, requires_grad=True)
        result = SO3_Mul.apply(x, y)
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
    parser = argparse.ArgumentParser(description="Benchmark SO3 Mul forward/backward")
    parser.add_argument("--mode", choices=["fwd", "bwd"], default="fwd", help="Benchmark forward or backward pass")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to run on")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "fp64"], default="fp32", help="Data type")
    parser.add_argument("--size", type=int, default=10000, help="Batch size")
    parser.add_argument("--kind", choices=["n-n", "1-n", "n-1"], default="n-n", help="Broadcasting pattern")
    args = parser.parse_args()
    
    dtype = DTYPE_MAP[args.dtype]
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Benchmarking SO3.Mul {args.mode} | device={args.device} | dtype={args.dtype} | size={args.size} | kind={args.kind}")
    print("-" * 80)
    
    if args.mode == "fwd":
        pp_bench, wp_bench = bench_forward(args.size, args.device, dtype, args.kind)
    else:
        pp_bench, wp_bench = bench_backward(args.size, args.device, dtype, args.kind)
    
    print(f"PyPose:  {pp_bench}")
    print(f"Warp:    {wp_bench}")
    print("-" * 80)
    
    speedup = pp_bench.median / wp_bench.median
    print(f"Speedup: {speedup:.2f}x {'(Warp faster)' if speedup > 1 else '(PyPose faster)'}")


if __name__ == "__main__":
    main()

