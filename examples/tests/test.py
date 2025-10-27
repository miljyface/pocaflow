import pocaflow as rs
import numpy as np

a = np.array([[1,2,3,4,5],[3,4,5,6,7]], dtype=np.float32)
b = np.array([[2,4,6],[8,10,12],[2,4,6],[8,10,12],[0,-1,2]], dtype=np.float32)

print(rs.matmul(a,b))