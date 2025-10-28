import time
import numpy as np
import pocaflow as rs
import torch


class MatmulBackend:
    def __init__(self, name, runner, use_gpu_resident=False):
        self.name = name
        self.runner = runner
        self.use_gpu_resident = use_gpu_resident

    def bench(self, a, b, iterations=100, warmup=10):
        # Warmup
        for _ in range(warmup):
            try:
                result = self.runner(a, b)
                if hasattr(result, 'cpu'):
                    result = result.cpu()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"WARMUP ERROR in {self.name}: {e}")
                return None
        
        # Benchmark
        times = []
        for _ in range(iterations):
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t0 = time.perf_counter()
                c = self.runner(a, b)
                
                # Force synchronization for GPU operations
                if hasattr(c, 'cpu'):
                    c = c.cpu()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            except Exception as e:
                print(f"RUN ERROR in {self.name}: {e}")
                return None
        
        return {
            "mean": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "std": np.std(times)
        }


def get_backends():
    backends = []
    
    # Rust GPU backend
    backends.append(MatmulBackend(
        "Rust GPU (with transfer)",
        lambda a, b: rs.matmul(a, b)
    ))
    
    if torch.cuda.is_available():
        # PyTorch with transfer (fair comparison to Rust)
        backends.append(MatmulBackend(
            "PyTorch CUDA (with transfer)",
            lambda a, b: torch.matmul(
                torch.from_numpy(a).to("cuda"),
                torch.from_numpy(b).to("cuda")
            ).cpu().numpy()
        ))
        
        # PyTorch GPU-resident (best case, no transfer)
        def pytorch_gpu_resident(a_gpu, b_gpu):
            return torch.matmul(a_gpu, b_gpu)
        
        backends.append(MatmulBackend(
            "PyTorch CUDA (GPU-resident, no transfer)",
            pytorch_gpu_resident,
            use_gpu_resident=True
        ))
    
    return backends


def main():
    print("=== Matrix Multiplication Benchmark ===")
    print("(100 iterations, measuring mean/min/max)\n")
    
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    
    for n in sizes:
        print(f"\n=== Size {n}x{n} ===")
        
        # CPU arrays for backends that need them
        a_cpu = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
        b_cpu = np.ascontiguousarray(np.random.rand(n, n), dtype=np.float32)
        
        # GPU tensors for GPU-resident benchmark
        if torch.cuda.is_available():
            a_gpu = torch.from_numpy(a_cpu).to("cuda")
            b_gpu = torch.from_numpy(b_cpu).to("cuda")
        
        for backend in get_backends():
            if backend.use_gpu_resident and torch.cuda.is_available():
                result = backend.bench(a_gpu, b_gpu)
            else:
                result = backend.bench(a_cpu, b_cpu)
            
            if result:
                print(f" {backend.name}:")
                print(f"   Mean: {result['mean']:.2f} ms | Min: {result['min']:.2f} ms | Max: {result['max']:.2f} ms")

main()
