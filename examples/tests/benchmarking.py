import time
import numpy as np
import pocaflow as rs
import sys
import torch
TORCH_AVAILABLE = True
MPS_AVAILABLE = torch.backends.mps.is_available()
METAL_AVAILABLE = hasattr(rs, 'metal_matmul_f32')

def benchmark_function(func, warmup=3, iterations=10):
    for _ in range(warmup):
        try:
            func()
        except Exception as e:
            return None
    
    times = []
    for _ in range(iterations):
        try:
            times.append(func())
        except Exception as e:
            return None
    
    if not times:
        return None
        
    times_ms = np.array(times) * 1000
    return {
        'mean': np.mean(times_ms),
        'std': np.std(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms)
    }


def benchmark_rust_cpu(n=2048):
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
    
    def _run():
        start = time.perf_counter()
        _ = rs.matmul_f32_cpu(a, b)  # Use f32 variant
        return time.perf_counter() - start
    
    return benchmark_function(_run)


# WARNING: THIS SHIT IS SLOW
def benchmark_rust_strassen(n=2048):
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
    
    def _run():
        start = time.perf_counter()
        _ = rs.strassen_matmul_f32(a, b)  # Use f32 variant
        return time.perf_counter() - start
    
    return benchmark_function(_run)


def benchmark_rust_metal(n=2048):
    if not METAL_AVAILABLE:
        return None
    
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
    
    def _run():
        start = time.perf_counter()
        _ = rs.matmul(a, b)
        return time.perf_counter() - start
    
    return benchmark_function(_run)


def benchmark_numpy(n=2048):
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    
    def _run():
        start = time.perf_counter()
        _ = np.dot(a, b)
        return time.perf_counter() - start
    
    return benchmark_function(_run)


def benchmark_torch_cpu(n=2048):
    if not TORCH_AVAILABLE:
        return None
    
    a = torch.randn(n, n, dtype=torch.float32, device="cpu")
    b = torch.randn(n, n, dtype=torch.float32, device="cpu")
    
    def _run():
        start = time.perf_counter()
        _ = torch.matmul(a, b)
        return time.perf_counter() - start
    
    return benchmark_function(_run)


def benchmark_torch_mps(n=2048):
    if not TORCH_AVAILABLE or not MPS_AVAILABLE:
        return None
    
    a = torch.randn(n, n, dtype=torch.float32, device="mps")
    b = torch.randn(n, n, dtype=torch.float32, device="mps")
    
    def _run():
        torch.mps.synchronize()
        start = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.mps.synchronize()
        return time.perf_counter() - start
    
    return benchmark_function(_run)


def format_speedup(baseline, current):
    if current is None or current <= 0:
        return "N/A"
    ratio = baseline / current
    if ratio >= 1.0:
        return f"{ratio:>6.2f}x faster"
    else:
        return f"{1/ratio:>6.2f}x slower"


def print_system_info():
    print("System Information:")
    print("-" * 80)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")
    
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {MPS_AVAILABLE}")
    else:
        print("PyTorch: Not installed")
    
    print(f"Metal GPU support (f32): {METAL_AVAILABLE}")
    print()


def main():
    try:
        n = int(input("Square Matrix Dimensions (e.g., 2048): "))
    except (ValueError, EOFError):
        print("Invalid input, using default size 2048")
        n = 2048
    
    warmup = 2
    iterations = 10
    
    print("\n" + "=" * 80)
    print(f"Matrix Multiplication Benchmark (f32): ({n}×{n}) @ ({n}×{n})")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("=" * 80)
    print()
    
    print_system_info()
    
    results = {}
    
    benchmarks = [
        #("rust_cpu", "Rust CPU (Accelerate BLAS f32)", benchmark_rust_cpu),
        #("rust_strassen", "Rust Strassen (f32)", benchmark_rust_strassen),
        ("rust_metal", "Rust Metal GPU (f32)", benchmark_rust_metal),
        #("numpy", "NumPy (f32)", benchmark_numpy),
        #("torch_cpu", "PyTorch CPU (f32)", benchmark_torch_cpu),
        ("torch_mps", "PyTorch MPS (f32)", benchmark_torch_mps),
    ]
    
    for key, name, func in benchmarks:
        print(f"Benchmarking {name}...")
        result = func(n)
        if result is not None:
            results[key] = result
        else:
            print(f"  → Skipped (not available)")
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (f32 precision)")
    print("=" * 80)
    print(f"{'Implementation':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'vs NumPy':<15}")
    print("-" * 80)
    
    numpy_mean = results.get('numpy', {}).get('mean', 1.0)
    
    display_order = [
        #'rust_cpu',
        #'rust_strassen',
        'rust_metal',
        #'numpy',
        #'torch_cpu',
        'torch_mps',
    ]
    
    display_names = {
        #'rust_cpu': 'Rust CPU (Accelerate f32)',
        #'rust_strassen': 'Rust Strassen (f32)',
        'rust_metal': 'Rust Metal GPU (f32)',
        #'numpy': 'NumPy (f32)',
        #'torch_cpu': 'PyTorch CPU (f32)',
        'torch_mps': 'PyTorch MPS (f32)',
    }
    
    for key in display_order:
        if key not in results:
            continue
        
        r = results[key]
        speedup_str = format_speedup(numpy_mean, r['mean'])
        print(f"{display_names[key]:<30} {r['mean']:>10.4f}   {r['std']:>10.4f}   {speedup_str}")
    
    print("-" * 80)
    
    print("\nStatistics:")
    print("-" * 80)
    for key in display_order:
        if key not in results:
            continue
        r = results[key]
        print(f"{display_names[key]:<30} Min: {r['min']:>8.4f} ms  Max: {r['max']:>8.4f} ms")
    
    print("\n" + "=" * 80)
    if results:
        fastest = min(results.items(), key=lambda x: x[1]['mean'])
        print(f"Fastest: {display_names[fastest[0]]} ({fastest[1]['mean']:.4f} ms)")
    print("=" * 80)

main()
