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

# Matrix sizes
m, k, n = 1024, 1024, 1024

a = np.asfortranarray(np.random.randn(m, k).astype(np.float32))
b = np.asfortranarray(np.random.randn(k, n).astype(np.float32))

# Compute on CPU, and ensure output is Fortran-contiguous (column-major)
c_ref = np.dot(a, b)
c_ref_f = np.asfortranarray(c_ref)  # force to F, just to be sure

print("[NPY] a (F-order) first 8:", a.flatten(order='F')[:8])
print("[NPY] b (F-order) first 8:", b.flatten(order='F')[:8])
print("[NPY] c (F-order) first 8:", c_ref_f.flatten(order='F')[:8])
print("c_ref is Fortran contiguous?", c_ref_f.flags.f_contiguous)
