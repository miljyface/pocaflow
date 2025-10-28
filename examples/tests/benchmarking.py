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
            except Exception as e:
                print(f"WARMUP ERROR in {self.name}: {e}")
                return None
        times = []
        for _ in range(iterations):
            try:
                t0 = time.perf_counter()
                c = self.runner(a, b)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            except Exception as e:
                print(f"RUN ERROR in {self.name}: {e}")
                return None
        return {"mean": np.mean(times)}

def get_backends():
    backends = [MatmulBackend("Rust GPU (rs.matmul)", lambda a, b: rs.matmul(a, b))]
    if torch.cuda.is_available():
        backends.append(MatmulBackend("PyTorch CUDA", lambda a, b: torch.matmul(
            torch.from_numpy(a).to("cuda"),
            torch.from_numpy(b).to("cuda")
        ).cpu().numpy()))
    return backends

def main():
    print("=== Matrix Multiplication Benchmark ===")
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    for n in sizes:
        print(f"\n=== Size {n}x{n} ===")
        a = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
        b = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
        for backend in get_backends():
            result = backend.bench(a, b)
            if result:
                print(f" {backend.name}: {result['mean']:.2f} ms")

if __name__ == "__main__":
    main()
