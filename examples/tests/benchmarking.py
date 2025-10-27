import time
import numpy as np
import pocaflow as rs
import sys
import platform
import torch
TORCH_AVAILABLE = True

def hasattr_mod(obj, attr):
    try:
        return hasattr(obj, attr)
    except Exception:
        return False

PYTORCH_ENABLE_MPS_FALLBACK = 0
MPS_AVAILABLE = TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
METAL_AVAILABLE = hasattr_mod(rs, 'metal_matmul_f32')
CUDA_AVAILABLE = hasattr_mod(rs, 'cuda_matmul_f32') and sys.platform.startswith("win")

class MatmulBackend:
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

    # Pretty print table (backend as columns, sizes as rows)
    print("\n==== Results Table (mean ms per run) ====")
    header = ["Size"] + [name for name in backend_results]
    print(" | ".join(f"{h:>18}" for h in header))
    print("-" * (20 * len(header)))
    for idx, n in enumerate(sizes):
        row = [f"{n:>18}"]
        for name in backend_results:
            mean = backend_results[name]["means"][idx]
            row.append(f"{mean:>18.2f}")
        print(" | ".join(row))

main()
