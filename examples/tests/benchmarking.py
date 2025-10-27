import time
import numpy as np
import pocaflow as rs
import torch

class MatmulBackend:
    def __init__(self, name, runner):
        self.name = name
        self.runner = runner

    def bench(self, a, b, iterations=10, warmup=3):
        for _ in range(warmup):
            try:
                self.runner(a, b)
            except Exception:
                return None
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            c = self.runner(a, b)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        stat = np.array(times)
        result = {
            "mean": np.mean(stat),
            "std": np.std(stat),
            "min": np.min(stat),
            "max": np.max(stat),
        }
        return result

def get_backends():
    backends = []
    # Rust GPU/Metal/CUDA generic entrypoint
    backends.append(MatmulBackend("Rust GPU/Metal/CUDA (rs.matmul)", lambda a, b: rs.cuda_matmul_f32(a, b)))
    # PyTorch CUDA/MPS
    if torch.cuda.is_available():
        backends.append(MatmulBackend("PyTorch CUDA (torch.matmul)", lambda a, b: torch.matmul(
            torch.from_numpy(a).to("cuda"),
            torch.from_numpy(b).to("cuda")
        ).cpu().numpy()))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append(MatmulBackend("PyTorch MPS (torch.matmul)", lambda a, b: torch.matmul(
            torch.from_numpy(a).to("mps"),
            torch.from_numpy(b).to("mps")
        ).cpu().numpy()))
    return backends

def main():
    print("=== Matrix Multiplication Benchmark (GPU only) ===")
    sizes = [256, 512, 1024, 2048, 4096]
    iterations = 10
    warmup = 3
    dtype = np.float32

    backend_results = {}
    backends = get_backends()
    for backend in backends:
        backend_results[backend.name] = {"sizes": [], "means": []}
    for n in sizes:
        print(f"\n=== Benchmarking size {n}x{n} ===")
        a = np.ascontiguousarray(np.random.rand(n, n), dtype=dtype)
        b = np.ascontiguousarray(np.random.rand(n, n), dtype=dtype)
        for backend in backends:
            print(f" {backend.name}...", end="", flush=True)
            result = backend.bench(a, b, iterations=iterations, warmup=warmup)
            if result:
                backend_results[backend.name]["sizes"].append(n)
                backend_results[backend.name]["means"].append(result["mean"])
                print(f" {result['mean']:.2f} ms")
            else:
                backend_results[backend.name]["sizes"].append(n)
                backend_results[backend.name]["means"].append(np.nan)
                print(f" (skipped)")

    print("\n==== Results Table (mean ms per run) ====")
    header = ["Size"] + [name for name in backend_results]
    print(" | ".join(f"{h:>24}" for h in header))
    print("-" * (26 * len(header)))
    for idx, n in enumerate(sizes):
        row = [f"{n:>24}"]
        for name in backend_results:
            mean = backend_results[name]["means"][idx]
            row.append(f"{mean:>24.2f}")
        print(" | ".join(row))

main()
