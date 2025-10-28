import torch
import numpy as np
import time
import pocaflow as pf

def benchmark(label, func, a, b, n_warmup=10, n_repeat=20):
    for _ in range(n_warmup):
        out = func(a, b)
        # Synchronize CUDA if result is a Torch tensor
        if isinstance(out, torch.Tensor):
            torch.cuda.synchronize()
        # Synchronize CUDA if input is Rust GPU tensor (you can add device sync logic here if needed)

    times = []
    for _ in range(n_repeat):
        if isinstance(a, torch.Tensor):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = func(a, b)
        if isinstance(out, torch.Tensor):  # Rust Tensor won't trigger here
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {'median': np.median(np.array(times) * 1000), 'output': out}

sizes = [(512, 512, 512), (2048, 2048, 2048), (8192, 8192, 8192)]

print("="*80)
print("GPU Tensor Benchmark: Rust (pf.Tensor) vs PyTorch (torch.Tensor)")
print("="*80)

for M, K, N in sizes:
    print(f"\n({M}x{K}) @ ({K}x{N})")

    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    # Rust GPU tensor API (copy only once)
    a_rust = pf.Tensor.from_array(a_np, device="cuda")
    b_rust = pf.Tensor.from_array(b_np, device="cuda")
    rust_result = benchmark("Rust", pf.matmul, a_rust, b_rust)

    # If result is GPU tensor, convert to numpy for accuracy check
    if hasattr(rust_result['output'], 'numpy'):
        c_rust_np = rust_result['output'].numpy()
    else:
        c_rust_np = np.array(rust_result['output'])

    # PyTorch
    a_torch = torch.from_numpy(a_np).cuda()
    b_torch = torch.from_numpy(b_np).cuda()
    torch_result = benchmark("PyTorch", lambda a, b: a @ b, a_torch, b_torch)
    c_torch_np = torch_result['output'].cpu().numpy()

    speedup = torch_result['median'] / rust_result['median']
    flops = 2.0 * M * K * N / 1e9
    gflops_rust = flops / (rust_result['median'] / 1000)
    gflops_torch = flops / (torch_result['median'] / 1000)
    error = np.max(np.abs(c_rust_np - c_torch_np))

    print(f"Rust:    {rust_result['median']:.2f} ms ({gflops_rust:.1f} GFLOPS)")
    print(f"PyTorch: {torch_result['median']:.2f} ms ({gflops_torch:.1f} GFLOPS)")
    print(f"Speedup (Torch/Rust): {speedup:.2f}x | Max error: {error:.2e}")

# Optional: compare with legacy NumPy API (pf.matmul with NumPy arrays) for a baseline
for M, K, N in sizes[:1]:
    print(f"\nNumPy fallback ({M}x{K}) @ ({K}x{N})")
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    c_np = pf.matmul(a_np, b_np)
    print(f"Legacy matmul result shape: {c_np.shape}, dtype: {c_np.dtype}")
