"""
GPU Matmul Benchmark: Rust cuBLAS-LT vs PyTorch CUDA
Optimized for RTX 4090 with Tensor Cores
"""
import torch
import numpy as np
import time
import pocaflow as pf

def benchmark(label, matmul_func, a, b, n_warmup=10, n_repeat=20):
    """Benchmark with extensive warmup for GPU"""
    for _ in range(n_warmup):
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor) and out.device.type == "cuda":
            torch.cuda.synchronize()
    
    times = []
    for _ in range(n_repeat):
        if isinstance(a, torch.Tensor) and a.device.type == "cuda":
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        out = matmul_func(a, b)
        
        if isinstance(out, torch.Tensor) and out.device.type == "cuda":
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - t0)
    
    times = np.array(times) * 1000
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'median': np.median(times),
        'output': out
    }

def run_benchmark():
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    print("="*80)
    print("GPU Matrix Multiplication Benchmark - cuBLAS-LT vs PyTorch")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print("="*80)
    
    results = []
    
    for M, K, N in sizes:
        print(f"\nMatrix Size: ({M}x{K}) @ ({K}x{N})")
        print("-"*80)
        
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        
        # Rust cuBLAS-LT
        rust_stats = benchmark("Rust cuBLAS-LT", pf.matmul, 
                              pf.Tensor.from_array(a_np), 
                              pf.Tensor.from_array(b_np))
        out_rust = rust_stats['output'].numpy()
        
        # PyTorch CUDA
        a_torch = torch.from_numpy(a_np).cuda()
        b_torch = torch.from_numpy(b_np).cuda()
        def torch_matmul(a, b): return (a @ b).cpu()
        
        torch_stats = benchmark("PyTorch CUDA", torch_matmul, a_torch, b_torch)
        out_torch = torch_stats['output'].numpy()
        
        # Results
        max_err = np.max(np.abs(out_rust - out_torch))
        speedup = torch_stats['median'] / rust_stats['median']
        
        flops = 2.0 * M * K * N
        gflops_rust = (flops / 1e9) / (rust_stats['median'] / 1000)
        gflops_torch = (flops / 1e9) / (torch_stats['median'] / 1000)
        
        print(f"Rust:    {rust_stats['median']:>7.2f} ms  "
              f"(±{rust_stats['std']:>5.2f}, {gflops_rust:>7.1f} GFLOPS)")
        print(f"PyTorch: {torch_stats['median']:>7.2f} ms  "
              f"(±{torch_stats['std']:>5.2f}, {gflops_torch:>7.1f} GFLOPS)")
        print(f"Speedup: {speedup:.2f}x | Error: {max_err:.2e}")
        
        results.append({
            'size': (M, K, N),
            'rust': rust_stats['median'],
            'torch': torch_stats['median'],
            'speedup': speedup,
            'error': max_err,
        })
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Size':<20} {'Rust (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-"*80)
    
    for r in results:
        size_str = f"{r['size'][0]}x{r['size'][1]}x{r['size'][2]}"
        print(f"{size_str:<20} {r['rust']:<12.2f} {r['torch']:<14.2f} {r['speedup']:<10.2f}")
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    print("="*80)
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print("="*80)

run_benchmark()
