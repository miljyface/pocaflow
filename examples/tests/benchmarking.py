import time
import numpy as np
import rust_linalg as rs
import torch

def benchmark_function(func, warmup=3, iterations=10):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        times.append(func())
    times_ms = np.array(times) * 1000
    return {
        'mean': np.mean(times_ms),
        'std': np.std(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms)
    }

def benchmark_rust_blas(n=2048):
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float64)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float64)
    def _run():
        start = time.perf_counter()
        _ = rs.matmul(a, b)
        return time.perf_counter() - start
    return benchmark_function(_run)

def benchmark_rust_strassen(n=2048):
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float64)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float64)
    def _run():
        start = time.perf_counter()
        _ = rs.strassen_matmul(a, b)
        return time.perf_counter() - start
    return benchmark_function(_run)

def benchmark_numpy(n=2048):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    def _run():
        start = time.perf_counter()
        _ = np.dot(a, b)
        return time.perf_counter() - start
    return benchmark_function(_run)

def benchmark_torch_cpu(n=2048):
    a = torch.randn(n, n, dtype=torch.float64, device="cpu")
    b = torch.randn(n, n, dtype=torch.float64, device="cpu")
    def _run():
        start = time.perf_counter()
        _ = torch.matmul(a, b)
        return time.perf_counter() - start
    return benchmark_function(_run)

def format_speedup(baseline, current):
    if current <= 0:
        return "N/A"
    ratio = baseline / current
    if ratio >= 1.0:
        return f"{ratio:>6.2f}x faster"
    else:
        return f"{1/ratio:>6.2f}x slower"

def test():
    n = int(input("Square Matrix Dimensions:"))
    warmup = 3
    iterations = 5
    
    print("="*80)
    print(f"Matrix Multiplication Benchmark: ({n}×{n}) @ ({n}×{n})")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("="*80)
    print()
    
    results = {}
    
    print("Benchmarking Rust BLAS...")
    results['rust_blas'] = benchmark_rust_blas(n)
    
    print("Benchmarking Rust Strassen...")
    results['rust_strassen'] = benchmark_rust_strassen(n)
    
    print("Benchmarking NumPy...")
    results['numpy'] = benchmark_numpy(n)
    
    print("Benchmarking PyTorch (CPU)...")
    results['torch_cpu'] = benchmark_torch_cpu(n)
    
    print()
    print("-"*80)
    print(f"{'Implementation':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'vs NumPy':<15}")
    print("-"*80)
    
    numpy_mean = results['numpy']['mean']
    order = ['rust_blas', 'rust_strassen', 'numpy', 'torch_cpu']
    
    for name in order:
        if name not in results:
            continue
        r = results[name]
        speedup_str = format_speedup(numpy_mean, r['mean'])
        print(f"{name:<20} {r['mean']:>10.4f}   {r['std']:>10.4f}   {speedup_str}")
    
    print("-"*80)
    print()
    print("Detailed Statistics:")
    print("-"*80)
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<20} Min: {r['min']:>8.4f} ms  Max: {r['max']:>8.4f} ms")
    print("="*80)

test()
