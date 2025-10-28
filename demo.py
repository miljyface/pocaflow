import pocaflow as pf
import numpy as np

print("===== pocaflow GPU matmul demo =====")
a = np.asfortranarray(np.random.randn(1024, 1024).astype(np.float32))
b = np.asfortranarray(np.random.randn(1024, 1024).astype(np.float32))

print("[NPY] a (F-order) first 8:", a.flatten(order='F')[:8])
print("[NPY] b (F-order) first 8:", b.flatten(order='F')[:8])

ta = pf.Tensor.from_array(a, device=0)  # device index = 0
tb = pf.Tensor.from_array(b, device=0)
tc = pf.matmul(ta, tb)
c = tc.numpy()
print("Max error vs NumPy:", np.abs(c - (a @ b)).max())