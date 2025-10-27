import pocaflow as rs
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return rs.dot(a,b)/(rs.magnitude(a)*rs.magnitude(b))

a = np.array([1,0], dtype=np.float64)
b = np.array([0,1], dtype=np.float64)
print(cosine(a,b))