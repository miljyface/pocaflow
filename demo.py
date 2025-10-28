import pocaflow as pf
import numpy as np

# Matrix sizes
m, k, n = 1024, 1024, 1024

# Always use Fortran-order for best performance!
a_np = np.asfortranarray(np.random.randn(m, k).astype(np.float32))
b_np = np.asfortranarray(np.random.randn(k, n).astype(np.float32))

# Create pocaflow tensors
a_tensor = pf.Tensor.from_array(a_np, device="cuda")
b_tensor = pf.Tensor.from_array(b_np, device="cuda")

# Perform cuBLAS-LT matmul
c_tensor = pf.matmul(a_tensor, b_tensor)

# Copy back to CPU
c_np = c_tensor.numpy()

# Validate against NumPy
c_ref = a_np @ b_np
print("Max error (should be <1e-4):", np.abs(c_np - c_ref).max())
