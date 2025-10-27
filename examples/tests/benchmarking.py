import time
import numpy as np
import pocaflow as rs
import sys
import platform
import torch
import matplotlib.pyplot as plt

TORCH_AVAILABLE = True

def hasattr_mod(obj, attr):
    try:
        return hasattr(obj, attr)
    except Exception:
        return False

PYTORCH_ENABLE_MPS_FALLBACK = 0
MPS_AVAILABLE = TORCH_AVAILABLE and torch.backends.mps.is_available()
METAL_AVAILABLE = hasattr_mod(rs, 'metal_matmul_f32')
CUDA_AVAILABLE = hasattr_mod(rs, 'cuda_matmul_f32') and sys.platform.startswith("win")

# --- snip: no changes to prompt_int or prompt_choice ---

class MatmulBackend:
    # ... no changes to this class ...
    def __init__(self, name, runner, validate_runner=None, enabled=True):
        self.name = name
        self.runner = runner
        self.enabled = enabled
        self.validate_runner = validate_runner if validate_runner is not None else runner

    def bench(self, a, b, iterations=10, warmup=3, validate=None):
        for _ in range(warmup):
            try:
                self.runner(a, b)
            except Exception:
                return None

        times = []
        c = None
        success = False 
        for _ in range(iterations):
            try:
                t0 = time.perf_counter()
                c = self.runner(a, b)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
                success = True
            except Exception:
                times.append(float('nan'))

        stat = np.array(times)
        result = {
            "mean": np.nanmean(stat),
            "std": np.nanstd(stat),
            "min": np.nanmin(stat),
            "max": np.nanmax(stat),
            "times": stat
        }

        if not success:
            result['rel_error'] = None
            return result

        if validate is not None:
            try:
                ref = validate(a, b)
                err = np.linalg.norm(c - ref) / np.linalg.norm(ref)
                result['rel_error'] = err
            except Exception:
                result['rel_error'] = None
        else:
            result['rel_error'] = None
        return result

def get_backends():
    # ... no changes to backend construction ...
    backends = [
        MatmulBackend("Rust CPU (f32)", lambda a, b: rs.matmul_f32_cpu(a, b)),
        MatmulBackend("NumPy (f32)", lambda a, b: np.dot(a, b)),
    ]
    if METAL_AVAILABLE:
        backends.append(MatmulBackend("Rust GPU (f32)", lambda a, b: rs.matmul(a, b)))
    if TORCH_AVAILABLE:
        backends.append(
            MatmulBackend(
                "PyTorch CPU (f32)",
                lambda a, b: torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).cpu().numpy()
            )
        )
        if MPS_AVAILABLE:
            backends.append(
                MatmulBackend(
                    "PyTorch GPU (f32)",
                    lambda a, b: torch.matmul(torch.from_numpy(a).to("mps"), torch.from_numpy(b).to("mps")).cpu().numpy()
                )
            )
    if CUDA_AVAILABLE:
        backends.append(
            MatmulBackend(
                "Rust CUDA (cuBLAS f32)",
                lambda a, b: rs.matmul(a, b)
            )
        )
    return backends

def print_system_info():
    print("System Information:")
    print("-" * 50)
    print(f"Platform:       {platform.system()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version:  {np.__version__}")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available:   {MPS_AVAILABLE}")
    print(f"Metal GPU support (f32): {METAL_AVAILABLE}")
    print(f"CUDA GPU support (f32):  {CUDA_AVAILABLE}")

def main():
    print("=== Matrix Multiplication Benchmark ===")
    print_system_info()

    sizes = [256, 512, 1024, 2048, 4096]
    iterations = 10
    warmup = 3
    dtype = np.float32
    validate = False

    backend_results = {}
    backends = get_backends()

    for backend in backends:
        backend_results[backend.name] = {"sizes": [], "means": []}

    for n in sizes:
        print(f"\n=== Benchmarking size {n}x{n} ===")
        a = np.ascontiguousarray(np.random.rand(n, n), dtype=dtype)
        b = np.ascontiguousarray(np.random.rand(n, n), dtype=dtype)
        ref_func = lambda a, b: np.dot(a, b)
        for backend in backends:
            print(f"  {backend.name}...", end="", flush=True)
            result = backend.bench(a, b, iterations=iterations, warmup=warmup, validate=(ref_func if validate else None))
            if result:
                backend_results[backend.name]["sizes"].append(n)
                backend_results[backend.name]["means"].append(result["mean"])
                print(f" {result['mean']:.2f} ms")
            else:
                backend_results[backend.name]["sizes"].append(n)
                backend_results[backend.name]["means"].append(np.nan)
                print(f" (skipped)")

    return backend_results, sizes

def plot_backend_results(backend_results, sizes):
    plt.figure(figsize=(12, 8))
    best_means = []
    best_labels = []

    # Plot each backend
    colors = plt.cm.tab10.colors
    for idx, (name, data) in enumerate(backend_results.items()):
        plt.plot(
            data["sizes"], data["means"],
            marker='o', label=name,
            color=colors[idx % len(colors)],
            linewidth=2 if idx == 0 else 1  # Optionally highlight first entry
        )

    # For each matrix size, find the lowest mean
    for i, n in enumerate(sizes):
        means_at_n = []
        labels_at_n = []
        for name, data in backend_results.items():
            means_at_n.append(data["means"][i])
            labels_at_n.append(name)
        min_time = min(means_at_n)
        min_idx = means_at_n.index(min_time)
        best_means.append(min_time)
        best_labels.append(labels_at_n[min_idx])
        # Annotate best
        plt.scatter(n, min_time, color='red', s=120, edgecolor='black', zorder=10)
        plt.text(
            n, min_time*1.1, f'Winner: {labels_at_n[min_idx]}', 
            color='black', ha='center', va='bottom', fontsize=10,
            fontweight='bold'
        )

    plt.xlabel("Matrix size (n)", fontsize=15)
    plt.ylabel("Mean time (ms)", fontsize=15)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.title("Matrix multiplication performance (mean time per backend)", fontsize=18, fontweight='bold')
    plt.legend(fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

backend_results, sizes = main()
plot_backend_results(backend_results, sizes)
