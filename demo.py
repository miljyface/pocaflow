import time
import numpy as np
import torch
import pocaflow as pf

def benchmark_matmul_tensor(libname, matmul_func, a, b, n_warmup=3, n_repeat=10):
    # Warmup
    for _ in range(n_warmup):
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor) and out.device.type == "mps":
            torch.mps.synchronize()

    timings = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        out = matmul_func(a, b)
        if isinstance(out, torch.Tensor) and out.device.type == "mps":
            torch.mps.synchronize()
        timings.append(time.perf_counter() - start)
    avg = np.mean(timings)
    print(f"[{libname}] avg time: {avg*1000:.2f} ms")
    return out, avg

# Try slightly smaller shapes on Mac Metal to avoid OOM
M, K, N = 8192, 8192, 8192

print("\n=== Benchmarking Large matmul (float32, Tensor API) ===")
a_np = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
b_np = np.random.uniform(-1, 1, (K, N)).astype(np.float32)

# Pocaflow Metal Tensor
a_pf = pf.Tensor.from_array(a_np)
b_pf = pf.Tensor.from_array(b_np)
def pocaflow_matmul_tensor(a, b):
    return pf.Tensor.from_array(pf.matmul(a.numpy(), b.numpy()))

out_pf, time_pf = benchmark_matmul_tensor("pocaflow(metal)", pocaflow_matmul_tensor, a_pf, b_pf)
out_pf_np = out_pf.numpy()

# PyTorch MPS or fallback
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Using PyTorch MPS backend.")
    a_torch = torch.from_numpy(a_np).to("mps")
    b_torch = torch.from_numpy(b_np).to("mps")
    def torch_matmul(a, b):
        return (a @ b).to("cpu")
else:
    print("MPS not available, using CPU.")
    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)
    def torch_matmul(a, b):
        return a @ b

out_torch, time_torch = benchmark_matmul_tensor("torch(mps/cpu)", torch_matmul, a_torch, b_torch)
out_torch_np = out_torch.numpy()

# Accuracy and speedup checks
max_err = np.max(np.abs(out_pf_np - out_torch_np))
print(f"Max absolute error (pocaflow vs torch): {max_err:.4g}")

speedup = time_torch / time_pf if time_pf > 0 else float('nan')
print(f"Speedup (torch/pocaflow): {speedup:.2f}x")

print("\n== Done ==")
