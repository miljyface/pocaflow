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

MPS_AVAILABLE = TORCH_AVAILABLE and torch.backends.mps.is_available()
METAL_AVAILABLE = hasattr_mod(rs, 'metal_matmul_f32')
CUDA_AVAILABLE = hasattr_mod(rs, 'cuda_matmul_f32') and sys.platform.startswith("win")

def prompt_int(prompt, default):
    s = input(prompt + f" [default {default}]: ")
    try:
        return int(s)
    except Exception:
        return default

def prompt_choice(prompt, choices, default):
    s = input(prompt + f" {choices} [default {default}]: ")
    s = s.strip().lower()
    if not s:
        return default
    opts = {c.lower(): c for c in choices}
    return opts.get(s, default)

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
        # makes sure c is always defined
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
        MatmulBackend("Rust CPU (Accelerate f32)", lambda a, b: rs.matmul_f32_cpu(a, b)),
        MatmulBackend("NumPy (f32)", lambda a, b: np.dot(a, b)),
    ]
    if METAL_AVAILABLE:
        backends.append(MatmulBackend("Rust Metal GPU (f32)", lambda a, b: rs.matmul(a, b)))
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
                    "PyTorch MPS (f32)",
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
    n = prompt_int("Square matrix size (n x n)", 2048)
    iterations = prompt_int("Iteration count", 10)
    validate = prompt_choice("Validate result accuracy? (can slow down)", ['no', 'yes'], 'no') == 'yes'
    dtype = np.float32

    print(f"Generating input matrices A, B of shape ({n}, {n}) with dtype {dtype}...")
    a = np.ascontiguousarray(np.random.rand(n, n), dtype=dtype)
    b = np.ascontiguousarray(np.random.rand(n, n), dtype=dtype)

    backends = get_backends()

    print("\n--- Running Benchmarks ---\n")
    results = {}
    ref_func = lambda a, b: np.dot(a, b)
    for backend in backends:
        print(f"Benchmarking {backend.name}...")
        result = backend.bench(a, b, iterations=iterations, warmup=3, validate=(ref_func if validate else None))
        if result:
            results[backend.name] = result
            print(f"   mean: {result['mean']:.4f} ms, std: {result['std']:.4f} ms, min: {result['min']:.4f} ms, max: {result['max']:.4f} ms" +
                  (f", rel err: {result['rel_error']:.2e}" if validate and 'rel_error' in result else ""))
        else:
            print("   Skipped (backend not available or errored)")

    print("\n===== RESULTS =====")
    print(f"{'Implementation':<28} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min':>12} {'Max':>12} {'RelErr':>10}")
    print("-" * 80)
    numpy_result = results.get("NumPy (f32)", {"mean": 1.0})
    base = numpy_result["mean"]
    for k, v in results.items():
        speedup = f"{(base/v['mean']):.2f}x faster" if v["mean"] > 0 and base else ""
        err_str = f"{v.get('rel_error', 0):.2e}" if validate and 'rel_error' in v else ""
        print(f"{k:<28} {v['mean']:12.4f} {v['std']:12.4f} {v['min']:12.4f} {v['max']:12.4f} {err_str:>10} {speedup}")

    print("-" * 80)
    best = min(results.items(), key=lambda x: x[1]['mean'])
    print(f"FASTEST: {best[0]} ({best[1]['mean']:.2f} ms)")

main()
