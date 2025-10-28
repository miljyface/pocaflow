import torch
import numpy as np
import time
import pocaflow as pf


def benchmark(label, matmul_func, a, b, n_warmup=5, n_repeat=10):
    """Run benchmark with warmup and timing"""
    # Warmup
    for _ in range(n_warmup):
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor) and out.device.type == "cuda":
            torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor) and out.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    times = np.array(times) * 1000  # convert to ms
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'output': out
    }


def run_benchmark_suite():
    """Run benchmarks for multiple matrix sizes"""
    
    # Matrix sizes to test
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192)
    ]
    
    print("="*70)
    print("GPU Matrix Multiplication Benchmark")
    print("Rust CUDA vs PyTorch CUDA")
    print("="*70)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires CUDA.")
        return
    
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("="*70)
    
    results = []
    
    for M, K, N in sizes:
        print(f"\nMatrix Size: ({M}x{K}) @ ({K}x{N})")
        print("-"*70)
        
        # Generate random data
        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)
        
        # ===== Rust/pocaflow Tensor API =====
        # Create Tensor objects (using your new Tensor structure)
        a_rust = pf.Tensor.from_array(a_np)
        b_rust = pf.Tensor.from_array(b_np)
        
        def rust_matmul(a, b):
            # Use Tensor.numpy() to bridge to your matmul function
            return pf.Tensor.from_array(pf.matmul(a.numpy(), b.numpy()))
        
        rust_stats = benchmark("Rust CUDA", rust_matmul, a_rust, b_rust)
        out_rust_np = rust_stats['output'].numpy()
        
        # ===== PyTorch CUDA =====
        a_torch = torch.from_numpy(a_np).cuda()
        b_torch = torch.from_numpy(b_np).cuda()
        
        def torch_matmul(a, b):
            return (a @ b).cpu()
        
        torch_stats = benchmark("PyTorch CUDA", torch_matmul, a_torch, b_torch)
        out_torch_np = torch_stats['output'].numpy()
        
        # ===== Accuracy Check =====
        max_err = np.max(np.abs(out_rust_np - out_torch_np))
        rel_err = max_err / (np.max(np.abs(out_torch_np)) + 1e-10)
        
        # ===== Print Results =====
        print(f"Rust CUDA:    {rust_stats['mean']:>7.2f} ms  "
              f"(±{rust_stats['std']:>5.2f} ms, "
              f"min={rust_stats['min']:>6.2f}, max={rust_stats['max']:>6.2f})")
        
        print(f"PyTorch CUDA: {torch_stats['mean']:>7.2f} ms  "
              f"(±{torch_stats['std']:>5.2f} ms, "
              f"min={torch_stats['min']:>6.2f}, max={torch_stats['max']:>6.2f})")
        
        speedup = torch_stats['mean'] / rust_stats['mean']
        print(f"\nSpeedup (PyTorch/Rust): {speedup:.2f}x")
        print(f"Accuracy - Max abs error: {max_err:.2e}, Relative error: {rel_err:.2e}")
        
        # Compute theoretical FLOPS
        flops = 2.0 * M * K * N  # multiply-add = 2 FLOPs
        gflops_rust = (flops / 1e9) / (rust_stats['mean'] / 1000)
        gflops_torch = (flops / 1e9) / (torch_stats['mean'] / 1000)
        
        print(f"Performance - Rust: {gflops_rust:.2f} GFLOPS, "
              f"PyTorch: {gflops_torch:.2f} GFLOPS")
        
        results.append({
            'size': (M, K, N),
            'rust_mean': rust_stats['mean'],
            'torch_mean': torch_stats['mean'],
            'speedup': speedup,
            'max_err': max_err,
            'gflops_rust': gflops_rust,
            'gflops_torch': gflops_torch
        })
    
    # ===== Summary Table =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Size':<20} {'Rust (ms)':<12} {'PyTorch (ms)':<14} "
          f"{'Speedup':<10} {'Accuracy'}")
    print("-"*70)
    
    for r in results:
        size_str = f"{r['size'][0]}x{r['size'][1]}x{r['size'][2]}"
        print(f"{size_str:<20} {r['rust_mean']:<12.2f} {r['torch_mean']:<14.2f} "
              f"{r['speedup']:<10.2f} {r['max_err']:.2e}")
    
    print("="*70)
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print("="*70)


if __name__ == "__main__":
    run_benchmark_suite()
