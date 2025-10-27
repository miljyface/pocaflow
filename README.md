# Minimal-dependency Linear Algebra (pocaflow)

This is my personal playground for learning and building high-performance, low-level linear algebra in **Rust**, designed for speed, low dependencies, and compatibility with Python, running natively on Apple ARM64 (M1/M2/M3/MacBook Air, etc).

## Project Goals

- **Minimal dependencies**
- **Speed**: Leverage Apple Accelerate, BLAS, and custom cache-blocked and parallel routines to approach or beat NumPy/Torch for real matrix math—especially on ARM64.
- **Python interop**: Export core operations as Python bindings with [PyO3](https://pyo3.rs/), making it easy to drop-in as a NumPy accelerator.
- **Educational**: Code is written for clarity and control. Great for learning linalg, memory layout, and systems-level performance.

## Features

- Fast matrix multiplication (`matmul`), batched matmul, dot, cross, normalization, and more.
- Accept and return `numpy.ndarray` natively via PyO3.
- Pure Rust fallback kernels for ARM64 platforms with or without Accelerate (optimized for your hardware).
- Speed comparable to NumPy and PyTorch.

## Usage Examples

### From Python

```python
import pocaflow as rs
import numpy as np

a = np.random.rand(2048, 2048)
b = np.random.rand(2048, 2048)
c = rs.matmul(a, b)
print(c.shape)
```

### From Rust

```rust
use pocaflow::matmul;
let a = ndarray::Array2::<f64>::ones((2048, 2048));
let b = ndarray::Array2::<f64>::ones((2048, 2048));
let c = matmul(a.view(), b.view());
```

## Demos and Playground

- **Rotating ASCII cubes**

## Rust? Why not C?

- Control over memory layout, cache behavior, parallelism, and vectorization.
- I don't like C.
- I still had to use C for metal.
- I'll probably have to use it again for CUDA
- Fuck my life

## Roadmap

- More batched/parallel ops (e.g. broadcasting, batched dot/cross for 3D graphics).
- Deeper integration: autodiff, sparse/blending, machine learning stubs.
- Accessible ML Framework

## Contributing & Feedback

This project is **experimental**. Issues, PRs, and ideas are welcome.
