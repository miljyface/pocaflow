#!/usr/bin/env python3
import torch
import numpy as np
import time
import pocaflow as pf

print("="*70)
print("GPU Matmul Benchmark: Rust cuBLAS-LT vs PyTorch")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*70)

sizes = [(512, 512), (2048, 2048), (4096, 4096)]

for size in sizes:
    m, n = size
    k = size[0]
    
    print(f"\n[{m}x{k}] @ [{k}x{n}]")
    
    # Fresh data each iteration
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)
    
    # Rust: Fresh tensors
    a_rust = pf.Tensor.from_array(a_np, device="cuda")
    b_rust = pf.Tensor.from_array(b_np, device="cuda")
    
    # Warmup
    for _ in range(5):
        c_rust = pf.matmul(a_rust, b_rust)
    
    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        c_rust = pf.matmul(a_rust, b_rust)
        times.append(time.perf_counter() - t0)
    
    rust_time = np.median(times) * 1000
    c_np = c_rust.numpy()
    
    # PyTorch
    a_torch = torch.from_numpy(a_np).cuda()
    b_torch = torch.from_numpy(b_np).cuda()
    
    for _ in range(5):
        _ = a_torch @ b_torch
        torch.cuda.synchronize()
    
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c_torch = a_torch @ b_torch
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    torch_time = np.median(times) * 1000
    
    # Results
    flops = 2.0 * m * k * n / 1e9
    gflops_rust = flops / (rust_time / 1000)
    gflops_torch = flops / (torch_time / 1000)
    error = np.max(np.abs(c_np - c_torch.cpu().numpy()))
    
    print(f"Rust:    {rust_time:7.2f} ms  |  {gflops_rust:7.1f} GFLOPS")
    print(f"PyTorch: {torch_time:7.2f} ms  |  {gflops_torch:7.1f} GFLOPS")
    print(f"Speedup: {torch_time/rust_time:.2f}x  |  Error: {error:.2e}")

print("\n" + "="*70)
