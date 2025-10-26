import mathcube as rs
import numpy as np

a = np.array([[1,2],[3,4]], dtype=np.float32)
b = np.array([[0,1],[-1,0]], dtype=np.float32)

print(rs.matmul(a,b))