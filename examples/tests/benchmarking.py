import pocaflow as pf
import numpy as np
import time

print("=== Optimized Performance Benchmark ===\n")

for size in [256, 512, 1024, 2048, 4096]:
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        _ = pf.matmul(a, b)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        c = pf.matmul(a, b)
        times.append(time.perf_counter() - start)
    
    mean_time = np.mean(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    gflops = (2 * size**3) / (np.mean(times) * 1e9)
    
    print(f"{size}x{size}:")
    print(f"  Mean: {mean_time:.2f}ms | Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")
    print(f"  GFLOPS: {gflops:.1f}")
    print()
