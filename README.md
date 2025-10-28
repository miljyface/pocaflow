# Minimal-dependency Linear Algebra (pocaflow)

This is my personal playground for learning and building high-performance, low-level linear algebra in **Rust**, designed for speed, low dependencies, and compatibility with Python, running natively on Apple ARM64 (M1/M2/M3/MacBook Air, etc) **and now CUDA**.

## Project Goals

- **Minimal dependencies** — no extra baggage.
- **Speed**: Leverage Apple Accelerate, BLAS, and custom cache-blocked/parallel routines to approach or beat NumPy/Torch, especially on ARM64 and CUDA.
- **Python interop**: Core ops as Python bindings with [PyO3](https://pyo3.rs/), drop-in as a NumPy/Torch accelerator.
- **Educational**: Each kernel and buffer is explicit. Go deep into linalg, memory layout, GPU, and systems-level performance.
- **Enjoyment**: Where else can you write matrix multiplication by hand, yank tensor buffers, and scream at device memory errors?

## Features

- GPU matmul via CUDA/cuBLAS-LT (now actually correct).
- Fast CPU matmul (Accelerate, BLAS, pure Rust fallback).
- Accept and return `numpy.ndarray` natively via PyO3, handling Fortran vs C order properly.
- Device buffer pooling, explicit memory reuse, and (optional) manual sync for freaks who hate Python safety.
- Python demo scripts: matches NumPy/Torch with max error < 1e-4 (if not, fix your buffers).
- **Batched GPU matmul, device-to-device ops, and more soon.**

## How to Use

### Python

```python
import pocaflow as pf
import numpy as np

# Use Fortran-order arrays for GPU
a = np.asfortranarray(np.random.randn(1024, 1024).astype(np.float32))
b = np.asfortranarray(np.random.randn(1024, 1024).astype(np.float32))
ta = pf.Tensor.from_array(a, device="cuda")
tb = pf.Tensor.from_array(b, device="cuda")
tc = pf.matmul(ta, tb)
c = tc.numpy()

print("Max error vs NumPy:", np.abs(c - (a @ b)).max())  # Should be < 1e-4
```

### Rust

```rust
use pocaflow::matmul;
let a = ndarray::Array2::<f64>::ones((2048, 2048));
let b = ndarray::Array2::<f64>::ones((2048, 2048));
let c = matmul(a.view(), b.view());
```

## What I Learned (and Fixed)

- cuBLAS-LT is awesome, but easy to screw up: **memory layout, stride, and leading dimension must match exactly.**
- If you pass row-major, you get garbage. Use Fortran-order (column-major), and set descriptors to `(rows, cols, ld)` where `ld=rows`.
- Python interop: **validate shape/order and dtype at every step**. No silent bugs allowed.
- Zero dependencies, zero tolerance for silent misalignment: if max error > 1e-4, something still sucks.
- Device memory reuse is cool and fast, but you need to keep track, or you’ll corrupt results or crash.
- Every single matrix op must sanity-check shapes and error, or you’ll waste days screaming at RAM.
- Batching, streaming, and distributions in progress.

## Demos/Playground

- Minimal Python and Rust GPU/CPU matmul tests included.
- Rotating ASCII cubes (because, why not).

## Why Rust (and sometimes C)?

- Total control over memory, cache, parallelism, and vectorization.
- C is annoying and unsafe, but sometimes necessary for low-level driver access (Metal, CUDA).
- I'll use C only when the Rust ecosystem fails me.

## Current Roadmap

- [x] GPU-native matrix struct, PyO3 bindings
- [x] Matmul operates on device-resident buffers only
- [x] Python demo proves accuracy (see above)
- [ ] Remove all automatic synchronization
- [ ] Use pinned/page-locked memory for host buffers
- [ ] Workspace reuse for cuBLAS
- [ ] Batched matmul, high throughput benchmarking
- [ ] Multi-GPU and scalable device logic
- [ ] Broadcasting support
- [ ] Sparse matrices and neural network primitives

## Contributing & Feedback

This project is **experimental**. Issues, PRs, and ideas welcome.  
If you find a bug and fix it faster than me, **congrats** — send a PR.
