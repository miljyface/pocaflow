import time
import numpy as np

import rust_linalg as rs

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def bench(func, n, iters):
    times = []
    for _ in range(iters):
        times.append(func(n))
    return np.mean(times) * 1000  # ms

def benchmark_rs(n=1024):
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float64)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float64)
    _ = rs.matmul(a, b) # warmup
    start = time.time()
    _ = rs.matmul(a, b)
    return time.time() - start

def benchmark_numpy(n=1024):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    _ = np.dot(a, b) # warmup
    start = time.time()
    _ = np.dot(a, b)
    return time.time() - start

def benchmark_torch_cpu(n=1024):
    a = torch.randn((n, n), device="cpu")
    b = torch.randn((n, n), device="cpu")
    _ = torch.matmul(a, b)
    start = time.time()
    _ = torch.matmul(a, b)
    return time.time() - start

def benchmark_torch_mps(n=1024):
    a = torch.randn((n, n), device="mps")
    b = torch.randn((n, n), device="mps")
    _ = torch.matmul(a, b)
    torch.mps.synchronize()
    start = time.time()
    _ = torch.matmul(a, b)
    torch.mps.synchronize()
    return time.time() - start

def main():
    n = 2048     # matrix size (change as needed)
    iters = 10   # number of iterations for averaging
    print("-"*70)
    print("Implementation      Time (ms)      Speedup vs NumPy")
    print("-"*70)

    results = {}

    rust_time = bench(benchmark_rs, n, iters)
    results['rust_pure'] = rust_time

    # Numpy
    numpy_time = bench(benchmark_numpy, n, iters)
    results['numpy'] = numpy_time

    # Torch CPU
    torch_cpu_time = None
    if TORCH_AVAILABLE:
        torch_cpu_time = bench(benchmark_torch_cpu, n, iters)
        results['torch_cpu'] = torch_cpu_time

    # Torch MPS
    torch_mps_time = None
    if TORCH_AVAILABLE:
        try:
            torch.ones(1, device="mps")
            torch_mps_time = bench(benchmark_torch_mps, n, iters)
            results['torch_mps'] = torch_mps_time
        except Exception:
            torch_mps_time = None

    implementations = [
        ("rust_pure", results.get("rust_pure")),
        ("torch_cpu", results.get("torch_cpu")),
        ("numpy", results.get("numpy")),
        ("torch_mps", results.get("torch_mps")),
    ]

    numpy_ref = results["numpy"]

    for (name, t) in implementations:
        if t is None:
            continue
        speedup = numpy_ref / t if t > 0 else float('inf')
        if name == "numpy":
            print(f"{name:<18}{t:9.4f} ms   {speedup:>7.2f}x slower")
        elif speedup > 1:
            print(f"{name:<18}{t:9.4f} ms   {speedup:>7.2f}x faster")
        else:
            print(f"{name:<18}{t:9.4f} ms   {1/speedup:>7.2f}x slower")

    print("\n" + "-"*70)
    print(f"Matrix size: A({n}x{n}) @ B({n}x{n}) = C({n}x{n})")
    print(f"Iterations: {iters}")
    print("-"*70)

main()
