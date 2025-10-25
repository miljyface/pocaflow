import numpy as np
import torch
import time
from typing import Tuple
import rust_linalg

def init(
    size: Tuple[int, int, int],
    n_iterations: int = 100,
    warmup: int = 10
) -> dict:
    m, k, n = size
    
    # Generate random matrices (float32 for MPS compatibility)
    np_a = np.random.randn(m, k).astype(np.float32)
    np_b = np.random.randn(k, n).astype(np.float32)
    
    torch_a = torch.from_numpy(np_a)
    torch_b = torch.from_numpy(np_b)
    
    # For Apple Silicon Macs with MPS
    if torch.backends.mps.is_available():
        torch_a_mps = torch_a.to('mps')
        torch_b_mps = torch_b.to('mps')
    
    results = {}
    
    # --- NumPy Baseline ---

    for _ in range(warmup):
        _ = np_a @ np_b
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_np = np_a @ np_b
    end = time.perf_counter()
    results['numpy'] = (end - start) / n_iterations
    
    # --- PyTorch CPU ---
    for _ in range(warmup):
        _ = torch_a @ torch_b
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_torch = torch_a @ torch_b
    end = time.perf_counter()
    results['torch_cpu'] = (end - start) / n_iterations
    
    # --- PyTorch MPS (GPU on Apple Silicon) ---
    if torch.backends.mps.is_available():
        for _ in range(warmup):
            _ = torch_a_mps @ torch_b_mps
            torch.mps.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            result_mps = torch_a_mps @ torch_b_mps
            torch.mps.synchronize()
        end = time.perf_counter()
        results['torch_gpu'] = (end - start) / n_iterations
    
    # --- Rust (float32 version) ---
    for _ in range(warmup):
        _ = rust_linalg.matmul_f32(np_a, np_b)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        result_rust = rust_linalg.matmul_f32(np_a, np_b)
    end = time.perf_counter()
    results['rust_pure'] = (end - start) / n_iterations
    
    # Verify correctness
    rust_result = rust_linalg.matmul_f32(np_a, np_b)
    np.testing.assert_allclose(result_np, rust_result, rtol=1e-2, atol=1e-3)
    print(f"✓ Results verified for size {size}")
    
    return results

def compareSpeed(baseline: float, other: float) -> str:
    """Format speedup comparison."""
    if other < baseline:
        return f"{baseline/other:.2f}x faster"
    else:
        return f"{other/baseline:.2f}x slower"
    
# Test different matrix sizes
test_sizes = [
    (64, 64, 64, 1000),      # Small
    (256, 256, 256, 200),    # Medium
    (512, 512, 512, 50),     # Large
    (1024, 1024, 1024, 10),  # Very Large
    (2048, 2048, 2048, 5),   # Huge
    (4096, 4096, 4096, 1)    # Massive)
]
    
print(f"System Info:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  MPS available: {torch.backends.mps.is_available()}")
print(f"  NumPy version: {np.__version__}")
print(f"  Data type: float32")
print()
    
for m, k, n, iters in test_sizes:
    print(f"\n{'-'*80}")
    print(f"Matrix size: A({m}×{k}) @ B({k}×{n}) = C({m}×{n})")
    print(f"Iterations: {iters}")
    print(f"{'-'*80}")
        
    results = init((m, k, n), n_iterations=iters)

    # Sort by speed
    sorted_results = sorted(results.items(), key=lambda x: x[1])
       
    print(f"\n{'Method':<20} {'ms':<15} {'Comparison against Numpy':<20}")
    print("-" * 80)
        
    baseline = results['numpy']
    for impl, time_s in sorted_results:
        time_ms = time_s * 1000
        speedup = compareSpeed(baseline, time_s)
        print(f"{impl:<20} {time_ms:>10.4f} ms    {speedup:<20}")
    
print("\n" + "-" * 80)
print("Test complete")
print("-" * 80)
