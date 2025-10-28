import pocaflow as pf
import numpy as np
import torch
import time

print("=== Performance Benchmark ===\n")

for size in [256, 512, 1024, 2048, 4096, 8192]:
    print(f"{size}x{size}:")
    # Prepare inputs
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # Warmup (Rust/CUDA)
    for _ in range(5):
        _ = pf.matmul(a, b)
    
    # Benchmark Rust/CUDA
    times_rust = []
    for _ in range(10):
        start = time.perf_counter()
        c_rust = pf.matmul(a, b)
        times_rust.append(time.perf_counter() - start)
    mean_time_rust = np.mean(times_rust) * 1000
    min_time_rust = np.min(times_rust) * 1000
    max_time_rust = np.max(times_rust) * 1000
    gflops_rust = (2 * size**3) / (np.mean(times_rust) * 1e9)
    print(f"  Rust GPU:   Mean {mean_time_rust:.2f}ms | Min {min_time_rust:.2f}ms | Max {max_time_rust:.2f}ms | GFLOPS {gflops_rust:.1f}")

    # Torch (GPU) setup
    a_torch = torch.from_numpy(a).cuda()
    b_torch = torch.from_numpy(b).cuda()

    # Warmup
    for _ in range(5):
        torch.matmul(a_torch, b_torch)
        torch.cuda.synchronize()
    
    # Benchmark Torch CUDA
    times_torch = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        c_torch = torch.matmul(a_torch, b_torch)
        torch.cuda.synchronize()
        times_torch.append(time.perf_counter() - start)
    mean_time_torch = np.mean(times_torch) * 1000
    min_time_torch = np.min(times_torch) * 1000
    max_time_torch = np.max(times_torch) * 1000
    gflops_torch = (2 * size**3) / (np.mean(times_torch) * 1e9)
    print(f"  Torch CUDA: Mean {mean_time_torch:.2f}ms | Min {min_time_torch:.2f}ms | Max {max_time_torch:.2f}ms | GFLOPS {gflops_torch:.1f}")

    print()
