import rust_linalg as rs
import numpy as np

# Create input arrays
a = np.array([[1,2,4],[4,5,6],[7,8,9],[10,11,12],[13,14,15]], dtype=np.float64)  # shape (4, 3)
b = np.array([[9,2,5],[2,1,5],[4,0,1]], dtype=np.float64)  # shape (3, 3)

# Call the Rust matmul function
result = rs.matmul(a, b)

print(result)
