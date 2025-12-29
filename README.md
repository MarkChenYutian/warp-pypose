# warp-pypose

<p align="center">
  <strong>⚡ NVIDIA Warp-accelerated Lie group operations for PyPose</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-operations">Operations</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#development">Development</a>
</p>

<img width="3147" height="1049" alt="Log" src="https://github.com/user-attachments/assets/0ed05c38-d151-4502-aa17-b031b6bfab12" />

---

**warp-pypose** provides a high-performance [NVIDIA Warp](https://github.com/NVIDIA/warp)-based backend for [PyPose](https://github.com/pypose/pypose) LieTensor operations. It offers significant speedups for Lie group computations on both CPU and CUDA, with full support for automatic differentiation.

## Features

- 🚀 **Drop-in acceleration** — Seamlessly swap PyPose backends with a single function call
- ⚡ **Warp-powered kernels** — Optimized parallel implementations for CPU and CUDA
- 🔄 **Full autodiff support** — Analytical gradients for all operations with PyTorch integration
- 📐 **Comprehensive Lie group coverage** — SE(3), SO(3), se(3), so(3) algebras and groups
- 🎯 **FP16/FP32/FP64 precision** — Multi-precision support with numerically stable implementations
- 📊 **Arbitrary batch dimensions** — Full broadcasting support up to 4D batches

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU acceleration)

### Install from source

```bash
git clone https://github.com/MAC-VO/warp-pypose.git
cd warp-pypose
pip install -e .
```

### Dependencies

```bash
pip install torch pypose warp-lang
```

## Quick Start

### Basic Usage

```python
import torch
import pypose as pp
import pypose_warp

# Create a standard PyPose SE3 LieTensor
poses = pp.randn_SE3(1000, device="cuda", dtype=torch.float32)
points = torch.randn(1000, 3, device="cuda", dtype=torch.float32)

# Convert to Warp backend for accelerated computation
poses_warp = pypose_warp.to_warp_backend(poses)

# Use exactly like PyPose — all operations are accelerated
transformed = poses_warp.Act(points)      # Apply SE3 to points
matrices = poses_warp.matrix()            # Convert to 4x4 matrices
logs = poses_warp.Log()                   # Logarithm map to se3
composed = poses_warp @ poses_warp.Inv()  # Compose transformations
```

### Gradient Computation

```python
import torch
import pypose as pp
from pypose_warp import to_warp_backend

# Enable gradients
poses = pp.randn_SE3(100, device="cuda", requires_grad=True)
poses_warp = to_warp_backend(poses)
points = torch.randn(100, 3, device="cuda", requires_grad=True)

# Forward pass with Warp backend
result = poses_warp.Act(points)

# Backward pass — analytical gradients computed via Warp kernels
loss = result.sum()
loss.backward()

# Gradients available on original tensors
print(poses.grad.shape)   # (100, 7)
print(points.grad.shape)  # (100, 3)
```

### Backend Conversion

```python
from pypose_warp import to_warp_backend, to_pypose_backend, is_warp_backend

# Check and convert backends
poses = pp.randn_SE3(100)

if not is_warp_backend(poses):
    poses = to_warp_backend(poses)  # Convert to Warp for speed

# Convert back to PyPose if needed (e.g., for unsupported operations)
poses = to_pypose_backend(poses)
```

## Supported Operations

### SE(3) Group — Rigid Body Transformations

| Operation | Description | Method |
|-----------|-------------|--------|
| **Act** | Apply transform to 3D points | `X.Act(p)` |
| **Act4** | Apply transform to homogeneous points | `X.Act(p)` (4D) |
| **Mul** | Compose two SE3 transforms | `X @ Y` |
| **Inv** | Invert transformation | `X.Inv()` |
| **Log** | Logarithm map to se(3) | `X.Log()` |
| **Adj** | Adjoint action on se(3) | `X.Adj(a)` |
| **AdjT** | Transpose adjoint action | `X.AdjT(a)` |
| **Jinvp** | Inverse left Jacobian action | `X.Jinvp(p)` |
| **matrix** | Convert to 4×4 matrix | `X.matrix()` |
| **add_** | In-place update via Exp | `X.add_(delta)` |

### SO(3) Group — 3D Rotations

| Operation | Description | Method |
|-----------|-------------|--------|
| **Act** | Rotate 3D points | `R.Act(p)` |
| **Act4** | Rotate homogeneous points | `R.Act(p)` (4D) |
| **Mul** | Compose rotations | `R @ S` |
| **Log** | Logarithm map to so(3) | `R.Log()` |
| **Adj** | Adjoint action on so(3) | `R.Adj(a)` |
| **AdjT** | Transpose adjoint action | `R.AdjT(a)` |
| **Jinvp** | Inverse left Jacobian action | `R.Jinvp(p)` |
| **matrix** | Convert to 3×3 matrix | `R.matrix()` |
| **add_** | In-place update via Exp | `R.add_(delta)` |

### se(3) Algebra — SE(3) Tangent Space

| Operation | Description | Method |
|-----------|-------------|--------|
| **Exp** | Exponential map to SE(3) | `xi.Exp()` |
| **Mat** | Twist to 4×4 matrix | `xi.matrix()` |

### so(3) Algebra — SO(3) Tangent Space

| Operation | Description | Method |
|-----------|-------------|--------|
| **Exp** | Exponential map to SO(3) | `w.Exp()` |
| **Mat** | Angular velocity to 3×3 matrix | `w.matrix()` |
| **Jr** | Right Jacobian | `w.Jr()` |

## Benchmarks

Run the benchmark suite to compare Warp vs PyPose performance:

```bash
# Run all benchmarks (generates PNG charts)
python -m bench

# Run specific operator benchmarks
python -m bench.SE3_group
python -m bench.SO3_group
python -m bench.SE3_algebra
python -m bench.SO3_algebra

# Run individual operator with custom settings
python -m bench.SE3_group.Act --device cuda --dtype fp32 --size 10000
```

Benchmarks test across:
- **Devices**: CPU, CUDA
- **Data types**: FP16, FP32, FP64  
- **Batch sizes**: 128 to 32,768
- **Modes**: Forward and backward passes

Results are saved as PNG charts in the respective benchmark directories.

## Development

### Docker Environment

The recommended development environment uses Docker with NVIDIA GPU support:

```bash
# Auto-detect CUDA version and start container
./launch.sh

# Force specific CUDA version
FORCE_CUDA=12 ./launch.sh

# Mount additional paths
./launch.sh /path/to/dataset /path/to/models
```

Supported configurations:
- **Linux x86_64**: CUDA 12.x, CUDA 13.x
- **Jetson Orin**: CUDA 12.x (aarch64)
- **Jetson Thor**: CUDA 13.x (aarch64)

### Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_SE3_group_Act.py -v

# Run with specific device/dtype
pytest tests/ -v -k "cuda and fp32"

# Run with coverage
pytest tests/ --cov=pypose_warp --cov-report=html
```

### Project Structure

```
warp-pypose/
├── pypose_warp/
│   ├── __init__.py          # Backend conversion utilities
│   ├── ltype/
│   │   ├── SE3_group/        # SE(3) Lie group operations
│   │   ├── SO3_group/        # SO(3) Lie group operations
│   │   ├── SE3_algebra/      # se(3) Lie algebra operations
│   │   ├── SO3_algebra/      # so(3) Lie algebra operations
│   │   └── common/           # Shared kernel utilities
│   └── utils/
├── bench/                    # Benchmark suite
│   ├── SE3_group/
│   ├── SO3_group/
│   ├── SE3_algebra/
│   └── SO3_algebra/
├── tests/                    # Comprehensive test suite
├── docker/                   # Docker development environment
└── launch.sh                 # Container launch script
```

### Adding New Operations

Each operator follows a consistent pattern:

1. **Forward kernel** (`fwd.py`): Warp kernel implementing the operation
2. **Backward kernel** (`bwd.py`): Warp kernel for analytical gradients
3. **Autograd wrapper** (`__init__.py`): PyTorch Function connecting both

Example structure for `SE3_Act`:

```python
# fwd.py - Forward pass
@wp.kernel
def se3_act_kernel(...):
    # Warp kernel implementation
    
def SE3_Act_fwd(X, p):
    # Prepare tensors, launch kernel, return result

# bwd.py - Backward pass  
@wp.kernel
def se3_act_bwd_kernel(...):
    # Gradient computation kernel

def SE3_Act_bwd(X, out, grad_output):
    # Compute gradients

# __init__.py - PyTorch integration
class SE3_Act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, p):
        return SE3_Act_fwd(X, p)
    
    @staticmethod
    def backward(ctx, grad_output):
        return SE3_Act_bwd(...)
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyPose](https://github.com/pypose/pypose) — Differentiable Lie groups for robotics
- [NVIDIA Warp](https://github.com/NVIDIA/warp) — High-performance simulation and graphics programming

