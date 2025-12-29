"""
Automated benchmark runner for SE3 group operators.

Generates PNG charts comparing Warp vs PyPose backend performance across:
- Devices: CPU and CUDA
- Sizes: 128, 512, 2048, 8192, 32768
- Dtypes: fp16, fp32, fp64
- Modes: forward and backward

Each operator gets a separate PNG with multiple subplots.

Usage:
    python -m bench.SE3_group
"""
import os
import sys

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import torch
import warp as wp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Literal
import warnings

# Suppress some warnings during benchmarking
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize warp with minimal verbosity
wp.init()
wp.config.quiet = True

# Import benchmark modules
from . import Act, Act4, Inv, Mul, AdjXa, AdjTXa, Jinvp, Mat, Log, AddExp


# ============================================================================
# Configuration
# ============================================================================

SIZES = [128, 512, 2048, 8192, 32768]
DTYPES = ["fp16", "fp32", "fp64"]
DEVICES = ["cpu", "cuda"]
MODES = ["fwd", "bwd"]

# Colors
WARP_COLOR = "#85b737"
PYPOSE_COLOR = "#3070b7"

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

# Output directory
OUTPUT_DIR = Path(__file__).parent


# ============================================================================
# Operator Definitions
# ============================================================================

@dataclass
class OperatorInfo:
    """Information about an operator for benchmarking."""
    name: str
    bench_forward: Callable
    bench_backward: Callable
    has_kind: bool = False
    kind: str = "n-n"  # Default kind for operators that support it


# Define all operators
OPERATORS = [
    OperatorInfo(
        name="Act",
        bench_forward=Act.bench_forward,
        bench_backward=Act.bench_backward,
        has_kind=True,
    ),
    OperatorInfo(
        name="Act4",
        bench_forward=Act4.bench_forward,
        bench_backward=Act4.bench_backward,
        has_kind=True,
    ),
    OperatorInfo(
        name="Inv",
        bench_forward=Inv.bench_forward,
        bench_backward=Inv.bench_backward,
        has_kind=False,
    ),
    OperatorInfo(
        name="Mul",
        bench_forward=Mul.bench_forward,
        bench_backward=Mul.bench_backward,
        has_kind=True,
    ),
    OperatorInfo(
        name="AdjXa",
        bench_forward=AdjXa.bench_forward,
        bench_backward=AdjXa.bench_backward,
        has_kind=False,
    ),
    OperatorInfo(
        name="AdjTXa",
        bench_forward=AdjTXa.bench_forward,
        bench_backward=AdjTXa.bench_backward,
        has_kind=False,
    ),
    OperatorInfo(
        name="Jinvp",
        bench_forward=Jinvp.bench_forward,
        bench_backward=Jinvp.bench_backward,
        has_kind=False,
    ),
    OperatorInfo(
        name="Mat",
        bench_forward=Mat.bench_forward,
        bench_backward=Mat.bench_backward,
        has_kind=False,
    ),
    OperatorInfo(
        name="Log",
        bench_forward=Log.bench_forward,
        bench_backward=Log.bench_backward,
        has_kind=False,
    ),
    OperatorInfo(
        name="AddExp",
        bench_forward=AddExp.bench_forward,
        bench_backward=AddExp.bench_backward,
        has_kind=False,
    ),
]


# ============================================================================
# Benchmarking Functions
# ============================================================================

@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    device: str
    dtype: str
    size: int
    mode: str
    pp_median: float  # PyPose median time in seconds
    wp_median: float  # Warp median time in seconds
    error: Optional[str] = None


def run_benchmark(op: OperatorInfo, device: str, dtype_str: str, size: int, mode: str) -> BenchResult:
    """Run a single benchmark configuration."""
    dtype = DTYPE_MAP[dtype_str]
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        return BenchResult(
            device=device,
            dtype=dtype_str,
            size=size,
            mode=mode,
            pp_median=0,
            wp_median=0,
            error="CUDA not available"
        )
    
    try:
        if mode == "fwd":
            if op.has_kind:
                pp_bench, wp_bench = op.bench_forward(size, device, dtype, op.kind)
            else:
                pp_bench, wp_bench = op.bench_forward(size, device, dtype)
        else:
            if op.has_kind:
                pp_bench, wp_bench = op.bench_backward(size, device, dtype, op.kind)
            else:
                pp_bench, wp_bench = op.bench_backward(size, device, dtype)
        
        return BenchResult(
            device=device,
            dtype=dtype_str,
            size=size,
            mode=mode,
            pp_median=pp_bench.median,
            wp_median=wp_bench.median,
        )
    except Exception as e:
        return BenchResult(
            device=device,
            dtype=dtype_str,
            size=size,
            mode=mode,
            pp_median=0,
            wp_median=0,
            error=str(e)
        )


def benchmark_operator(op: OperatorInfo) -> list[BenchResult]:
    """Run all benchmark configurations for an operator."""
    results = []
    total = len(DEVICES) * len(DTYPES) * len(SIZES) * len(MODES)
    count = 0
    
    for mode in MODES:
        for device in DEVICES:
            for dtype in DTYPES:
                for size in SIZES:
                    count += 1
                    print(f"  [{count}/{total}] {mode} | {device} | {dtype} | size={size}", end="", flush=True)
                    result = run_benchmark(op, device, dtype, size, mode)
                    if result.error:
                        print(f" - SKIPPED ({result.error})")
                    else:
                        speedup = result.pp_median / result.wp_median if result.wp_median > 0 else 0
                        print(f" - {speedup:.2f}x")
                    results.append(result)
    
    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def create_operator_plot(op: OperatorInfo, results: list[BenchResult]):
    """Create a PNG plot for an operator with all benchmark results."""
    
    # Layout: 
    # Rows: modes (2) = 2 rows
    # Cols: devices (2) x dtypes (3) = 6 columns
    # Each subplot contains grouped bars for all sizes
    n_rows = len(MODES)  # 2
    n_cols = len(DEVICES) * len(DTYPES)  # 6
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(n_cols * 3.5, n_rows * 3.5),
        squeeze=False
    )
    
    # Style configuration
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0d1117')
    
    # Create result lookup dictionary
    result_map = {}
    for r in results:
        key = (r.mode, r.device, r.dtype, r.size)
        result_map[key] = r
    
    # Bar configuration
    n_sizes = len(SIZES)
    bar_width = 0.35
    x_positions = range(n_sizes)
    
    # Plot each subplot
    for row_idx, mode in enumerate(MODES):
        for col_idx, (device, dtype) in enumerate([(d, t) for d in DEVICES for t in DTYPES]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#161b22')
            
            # Collect data for all sizes
            wp_times = []
            pp_times = []
            has_data = False
            
            for size in SIZES:
                key = (mode, device, dtype, size)
                result = result_map.get(key)
                
                if result is None or result.error:
                    wp_times.append(0)
                    pp_times.append(0)
                else:
                    wp_times.append(result.wp_median * 1000)  # Convert to ms
                    pp_times.append(result.pp_median * 1000)
                    has_data = True
            
            if not has_data:
                # No data - show placeholder
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       color='#8b949e', fontsize=12, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Plot grouped bars for all sizes
                x = [i - bar_width/2 for i in x_positions]
                warp_bars = ax.bar(
                    x, wp_times,
                    width=bar_width,
                    color=WARP_COLOR,
                    edgecolor='none',
                    label='Warp'
                )
                
                x = [i + bar_width/2 for i in x_positions]
                pypose_bars = ax.bar(
                    x, pp_times,
                    width=bar_width,
                    color=PYPOSE_COLOR,
                    edgecolor='none',
                    label='PyPose'
                )
                
                # Add value labels on bars
                for bars in [warp_bars, pypose_bars]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            # Format based on magnitude
                            if height >= 1:
                                label = f'{height:.1f}'
                            else:
                                label = f'{height:.2f}'
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height,
                                label,
                                ha='center', va='bottom',
                                fontsize=7, color='#c9d1d9',
                                rotation=45
                            )
                
                # X-axis: size labels
                ax.set_xticks(list(x_positions))
                ax.set_xticklabels([str(s) for s in SIZES], fontsize=8, color='#8b949e')
                ax.set_xlabel('Batch Size', fontsize=9, color='#8b949e')
                
                # Y-axis
                ax.tick_params(axis='y', labelsize=7, colors='#8b949e')
                ax.set_ylabel('Time (ms)', fontsize=9, color='#8b949e')
                
                # Grid
                ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#30363d')
                ax.set_axisbelow(True)
            
            # Title for each subplot
            title = f"{mode.upper()} | {device} | {dtype}"
            ax.set_title(title, fontsize=10, color='#c9d1d9', pad=6, fontweight='bold')
            
            # Style spines
            for spine in ax.spines.values():
                spine.set_color('#30363d')
                spine.set_linewidth(0.5)
    
    # Main title
    fig.suptitle(
        f'SE3 {op.name} Benchmark: Warp vs PyPose',
        fontsize=18, color='#f0f6fc', fontweight='bold', y=0.98
    )
    
    # Create legend
    legend_patches = [
        mpatches.Patch(color=WARP_COLOR, label='Warp Backend'),
        mpatches.Patch(color=PYPOSE_COLOR, label='PyPose Backend'),
    ]
    fig.legend(
        handles=legend_patches,
        loc='upper right',
        fontsize=10,
        framealpha=0.8,
        facecolor='#161b22',
        edgecolor='#30363d',
        labelcolor='#c9d1d9'
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = OUTPUT_DIR / f"{op.name}.png"
    fig.savefig(
        output_path,
        dpi=150,
        facecolor=fig.get_facecolor(),
        edgecolor='none',
        bbox_inches='tight'
    )
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run benchmarks for all operators and generate plots."""
    print("=" * 80)
    print("SE3 Group Operators Benchmark Suite")
    print("=" * 80)
    print(f"Sizes: {SIZES}")
    print(f"Dtypes: {DTYPES}")
    print(f"Devices: {DEVICES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available - skipping CUDA benchmarks")
    print("=" * 80)
    
    for op in OPERATORS:
        print(f"\nBenchmarking: {op.name}")
        print("-" * 40)
        
        results = benchmark_operator(op)
        
        print(f"\nGenerating plot for {op.name}...")
        create_operator_plot(op, results)
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

