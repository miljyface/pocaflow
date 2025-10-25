import rust_linalg as rs
from structure import tensor

c = tensor([[1,2],[3,4]])
b = tensor([[1,0],[0,1]])

# For vectors
x = tensor([1.0, 2.0, 3.0])
y = tensor([4.0, 5.0, 6.0])

def cosine(p, q):
    return rs.dot(p, q) / (rs.magnitude(p) * rs.magnitude(q))

print(cosine(x, y))  # Works directly with Tensor objects!

# Matrix multiplication also works
result = rs.matmul(c, b)
print(result)
