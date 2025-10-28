import torch
import numpy as np
import time
import sys
import pocaflow as pf

def benchmark(label, matmul_func, a, b, n_warmup=3, n_repeat=7):
    # Warmup
    for _ in range(n_warmup):
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor):
            dev = out.device.type
            if dev == "cuda":
                torch.cuda.synchronize()
            elif dev == "mps":
                torch.mps.synchronize()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor):
            dev = out.device.type
            if dev == "cuda":
                torch.cuda.synchronize()
            elif dev == "mps":
                torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    mean = np.mean(times)
    print(f'{label}: {mean*1000:.2f} ms')
    return out, mean

# Problem size
M, K, N = 1024, 1024, 1024

# Make random data
a_np = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
b_np = np.random.uniform(-1, 1, (K, N)).astype(np.float32)

# --- Rust backend (Metal or CUDA) ---
a_pf = pf.Tensor.from_array(a_np)
b_pf = pf.Tensor.from_array(b_np)

def rust_matmul(a, b):
    # If you have device selection, include here (e.g., pf.Tensor.matmul(a, b, device="cuda"))
    # Otherwise, default Metal on Mac, CUDA on Linux/Win
    return pf.Tensor.from_array(pf.matmul(a.numpy(), b.numpy()))

out_rust, time_rust = benchmark("Rust (Metal/CUDA)", rust_matmul, a_pf, b_pf)
out_rust_np = out_rust.numpy()

# --- Choose Torch backend dynamically ---
if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
    print("Torch backend: CUDA")
    a_torch = torch.from_numpy(a_np).cuda()
    b_torch = torch.from_numpy(b_np).cuda()
    def torch_matmul(a, b): return (a @ b).cpu()
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Torch backend: MPS")
    a_torch = torch.from_numpy(a_np).to("mps")
    b_torch = torch.from_numpy(b_np).to("mps")
    def torch_matmul(a, b): return (a @ b).to("cpu")
else:
    print("Torch backend: CPU (fallback, for reference only)")
    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)
    def torch_matmul(a, b): return a @ b

out_torch, time_torch = benchmark("Torch (CUDA/MPS/CPU)", torch_matmul, a_torch, b_torch)
out_torch_np = out_torch.numpy()

# --- Results ---
max_err = np.max(np.abs(out_rust_np - out_torch_np))
print(f"Max abs error (Rust vs Torch): {max_err:.4g}")
if time_rust > 0:
    print(f"Speedup (Torch/Rust): {time_torch / time_rust:.2f}x")
print("\n== Done ==")
