import torch
import numpy as np
import time
import pocaflow as pf

def benchmark(label, func, a, b, n_warmup=10, n_repeat=20):
    for _ in range(n_warmup):
        out = func(a, b)
        if isinstance(out, torch.Tensor):
            torch.cuda.synchronize()
    
    times = []
    for _ in range(n_repeat):
        if isinstance(a, torch.Tensor):
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        out = func(a, b)
        
        if isinstance(out, torch.Tensor):
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - t0)
    
    return {'median': np.median(np.array(times) * 1000), 'output': out}

sizes = [(512, 512, 512), (2048, 2048, 2048), (8192, 8192, 8192)]

print("="*80)
print("GPU-Native Tensor Benchmark: Rust vs PyTorch")
print("="*80)

for M, K, N in sizes:
    print(f"\n({M}x{K}) @ ({K}x{N})")
    
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    
    # Rust GPU tensors (copy once, keep on GPU)
    a_rust = pf.Tensor.from_array(a_np, device="cuda")
    b_rust = pf.Tensor.from_array(b_np, device="cuda")
    rust_result = benchmark("Rust", pf.matmul, a_rust, b_rust)
    
    # PyTorch
    a_torch = torch.from_numpy(a_np).cuda()
    b_torch = torch.from_numpy(b_np).cuda()
    torch_result = benchmark("PyTorch", lambda a, b: a @ b, a_torch, b_torch)
    
    speedup = torch_result['median'] / rust_result['median']
    flops = 2.0 * M * K * N / 1e9
    gflops_rust = flops / (rust_result['median'] / 1000)
    
    print(f"Rust:    {rust_result['median']:.2f} ms ({gflops_rust:.1f} GFLOPS)")
    print(f"PyTorch: {torch_result['median']:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
