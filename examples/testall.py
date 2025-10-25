import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import rust_linalg as rs
from library.structure import tensor
import time


def test_header(name):
    # pretty format text header
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}")


def test_matrix_multiplication():
    test_header("Matrix Multiplication Tests")
    
    print("\n1. Raw lists (f64):")
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    result = rs.matmul(a, b)
    print(f"  A shape: (2, 3)")
    print(f"  B shape: (3, 2)")
    print(f"  Result: {result}")
    assert len(result) == 2 and len(result[0]) == 2, "Shape mismatch"
    
    # Test with Tensor objects
    print("\n2. Tensor objects (f64):")
    A = tensor([[1.0, 2.0], [3.0, 4.0]])
    B = tensor([[5.0, 6.0], [7.0, 8.0]])
    result = rs.matmul(A, B)
    print(f"  A @ B = {result}")
    
    # Test f32 precision
    print("\n3. Single precision (f32):")
    a_f32 = [[1.0, 2.0], [3.0, 4.0]]
    b_f32 = [[2.0, 0.0], [0.0, 2.0]]
    result_f32 = rs.matmul_f32(a_f32, b_f32)
    print(f"  Result: {result_f32}")
    
    # Test larger matrices
    print("\n4. Larger matrix (5x3 @ 3x4):")
    a_large = [[float(i+j) for j in range(3)] for i in range(5)]
    b_large = [[float(i-j) for j in range(4)] for i in range(3)]
    result_large = rs.matmul(a_large, b_large)
    print(f"  Result shape: {len(result_large)}x{len(result_large[0])}")
    assert len(result_large) == 5 and len(result_large[0]) == 4
    
    print("✓ Matrix multiplication tests passed")


def test_vector_operations():
    test_header("Vector Operations Tests")
    
    # Dot product
    print("\n1. Dot product:")
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    dot_result = rs.dot(v1, v2)
    print(f"  {v1} · {v2} = {dot_result}")
    assert dot_result == 32.0, f"Expected 32.0, got {dot_result}"
    
    # With Tensor objects
    print("\n2. Dot product with Tensors:")
    x = tensor([1.0, 2.0, 3.0])
    y = tensor([4.0, 5.0, 6.0])
    dot_tensor = rs.dot(x, y)
    print(f"  Result: {dot_tensor}")
    
    # Cross product
    print("\n3. Cross product:")
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    cross_result = rs.cross(a, b)
    print(f"  {a} × {b} = {cross_result}")
    assert cross_result == [0.0, 0.0, 1.0], f"Cross product failed"
    
    # Another cross product
    print("\n4. Cross product (general vectors):")
    a = [2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0]
    cross_result = rs.cross(a, b)
    print(f"  {a} × {b} = {cross_result}")
    
    # Magnitude
    print("\n5. Vector magnitude:")
    v = [3.0, 4.0]
    mag = rs.magnitude(v)
    print(f"  |{v}| = {mag}")
    assert abs(mag - 5.0) < 1e-10, f"Expected 5.0, got {mag}"
    
    # Normalization
    print("\n6. Vector normalization:")
    v = [3.0, 4.0]
    normalized = rs.normalize(v)
    print(f"  normalize({v}) = {normalized}")
    norm_mag = rs.magnitude(normalized)
    print(f"  Magnitude of normalized: {norm_mag}")
    assert abs(norm_mag - 1.0) < 1e-10, f"Normalized magnitude not 1.0"
    
    # F32 versions
    print("\n7. Single precision operations:")
    v1_f32 = [1.0, 2.0, 3.0]
    v2_f32 = [4.0, 5.0, 6.0]
    dot_f32 = rs.dot_f32(v1_f32, v2_f32)
    print(f"  dot_f32: {dot_f32}")
    
    mag_f32 = rs.magnitude_f32([3.0, 4.0])
    print(f"  magnitude_f32: {mag_f32}")
    
    print("✓ Vector operations tests passed")


def test_batch_operations():
    """Test batch matrix multiplication"""
    test_header("Batch Operations Tests")
    
    # Batch matmul - stacked matrices (2D layout)
    print("\n1. Batch matrix multiplication (f64):")
    # Stack two 2x2 matrices as a 4x2 array
    a = [
        [1.0, 2.0], [3.0, 4.0],  # First 2x2 matrix
        [5.0, 6.0], [7.0, 8.0],  # Second 2x2 matrix
    ]
    b = [
        [1.0, 0.0], [0.0, 1.0],  # First 2x2 identity
        [2.0, 0.0], [0.0, 2.0],  # Second 2x2 scaled identity
    ]
    
    # Call with updated signature (no n parameter)
    results = rs.batch_matmul(a, b, batch_size=2, m=2, k=2)
    print(f"  Batch size: 2")
    print(f"  Matrix shape per batch: 2x2")
    print(f"  Results:")
    for i, mat in enumerate(results):
        print(f"    Batch {i}: {mat}")
    
    assert len(results) == 2, "Expected 2 batches"
    
    # With Tensor objects
    print("\n2. Batch with Tensors:")
    A_batch = tensor([
        [1.0, 2.0], [3.0, 4.0],
        [5.0, 6.0], [7.0, 8.0],
    ])
    B_batch = tensor([
        [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [0.0, 1.0],
    ])
    results_tensor = rs.batch_matmul(A_batch, B_batch, batch_size=2, m=2, k=2)
    print(f"  Results: {len(results_tensor)} matrices")
    for i, mat in enumerate(results_tensor):
        print(f"    Batch {i}: {mat}")
    
    # Single precision batch
    print("\n3. Batch multiplication (f32):")
    results_f32 = rs.batch_matmul_f32(a, b, batch_size=2, m=2, k=2)
    print(f"  f32 Results: {len(results_f32)} matrices")
    
    # Strided batch (shared B matrix)
    print("\n4. Strided batch multiplication:")
    a_strided = [
        [1.0, 2.0], [3.0, 4.0],
        [5.0, 6.0], [7.0, 8.0],
    ]
    b_shared = [[2.0, 0.0], [0.0, 2.0]]
    
    # Updated signature (no k, n parameters)
    results_strided = rs.strided_batch_matmul(a_strided, b_shared, batch_size=2, m=2)
    print(f"  Shared B matrix: {b_shared}")
    for i, mat in enumerate(results_strided):
        print(f"    Result {i}: {mat}")
    
    # Strided f32
    print("\n5. Strided batch (f32):")
    results_strided_f32 = rs.strided_batch_matmul_f32(a_strided, b_shared, batch_size=2, m=2)
    print(f"  f32 Results: {len(results_strided_f32)} matrices")
    
    print(" Batch operations tests passed")


def test_edge_cases():
    test_header("Edge Cases & Error Handling")
    
    # dimension mismatch
    print("\n1. Matrix dimension mismatch:")
    try:
        a = [[1.0, 2.0]]
        b = [[1.0, 2.0, 3.0]]
        rs.matmul(a, b)
        print("FUCK,  Should have raised error")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")
    
    # vector length mismatch
    print("\n2. Vector length mismatch:")
    try:
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0, 3.0]
        rs.dot(v1, v2)
        print("FUCK,  Should have raised error")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")
    
    # cross product on non-3D vectors
    print("\n3. Cross product on non-3D vectors:")
    try:
        v1 = [1.0, 2.0]
        v2 = [3.0, 4.0]
        rs.cross(v1, v2)
        print("FUCK,  Should have raised error")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")
    
    # zero vec normalization
    print("\n4. Zero vector normalization:")
    try:
        v = [0.0, 0.0, 0.0]
        rs.normalize(v)
        print("FUCK,  Should have raised error")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")
    
    # dimension mismatch
    print("\n5. Batch dimension mismatch:")
    try:
        a = [[1.0, 2.0], [3.0, 4.0]]  # only 2x2, not 4x2
        b = [[1.0, 0.0], [0.0, 1.0]]
        rs.batch_matmul(a, b, batch_size=2, m=2, k=2)
        print("FUCK,   Should have raised error")
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")
    
    print("Edge case tests passed")


def test_cosine_similarity():
    test_header("Cosine Similarity Test")
    
    def cosine(p, q):
        return rs.dot(p, q) / (rs.magnitude(p) * rs.magnitude(q))
    
    # orthogonal
    print("\n1. Perpendicular vectors:")
    v1 = tensor([1.0, 0.0, 0.0])
    v2 = tensor([0.0, 1.0, 0.0])
    cos_sim = cosine(v1, v2)
    print(f"  Cosine similarity: {cos_sim}")
    assert abs(cos_sim) < 1e-10, "Should be 0 for perpendicular vectors"
    
    # parallel
    print("\n2. Parallel vectors:")
    v1 = tensor([1.0, 2.0, 3.0])
    v2 = tensor([2.0, 4.0, 6.0])
    cos_sim = cosine(v1, v2)
    print(f"  Cosine similarity: {cos_sim}")
    assert abs(cos_sim - 1.0) < 1e-10, "Should be 1 for parallel vectors"
    
    # A -> vs -A ->
    print("\n3. Opposite vectors:")
    v1 = tensor([1.0, 2.0, 3.0])
    v2 = tensor([-1.0, -2.0, -3.0])
    cos_sim = cosine(v1, v2)
    print(f"  Cosine similarity: {cos_sim}")
    assert abs(cos_sim + 1.0) < 1e-10, "Should be -1 for opposite vectors"
    
    # random vec test
    print("\n4. Arbitrary vectors:")
    v1 = tensor([3.0, 4.0, 5.0])
    v2 = tensor([1.0, 2.0, 1.0])
    cos_sim = cosine(v1, v2)
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    print("✓ Cosine similarity tests passed")


def test_performance():
    """Basic performance test"""
    test_header("Performance Test")
    
    print("\n1. Medium matrix multiplication (100x100):")
    n = 100
    a = [[float(i+j) for j in range(n)] for i in range(n)]
    b = [[float(i-j) for j in range(n)] for i in range(n)]
    
    start = time.time()
    result = rs.matmul(a, b)
    elapsed = time.time() - start
    
    print(f"  Matrix size: {n}x{n}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Result shape: {len(result)}x{len(result[0])}")
    
    print("\n2. Large vector operations (10,000 elements):")
    v1 = [float(i) for i in range(10000)]
    v2 = [float(i+1) for i in range(10000)]
    
    start = time.time()
    dot_result = rs.dot(v1, v2)
    elapsed = time.time() - start
    
    print(f"  Vector size: 10,000")
    print(f"  Dot product time: {elapsed*1000:.2f} ms")
    print(f"  Result: {dot_result:.2e}")
    
    print("\n3. Batch operations (10 batches of 50x50):")
    batch_size = 10
    m = 50
    k = 50
    # Create stacked matrices
    a_batch = [[float(i+j) for j in range(k)] for _ in range(batch_size) for i in range(m)]
    b_batch = [[float(i-j) for j in range(k)] for _ in range(batch_size) for i in range(k)]
    
    start = time.time()
    batch_results = rs.batch_matmul(a_batch, b_batch, batch_size=batch_size, m=m, k=k)
    elapsed = time.time() - start
    
    print(f"  Batch size: {batch_size}")
    print(f"  Matrix size per batch: {m}x{k}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Results: {len(batch_results)} matrices")
    
    print("Performance tests completed")


def main():
    try:
        test_matrix_multiplication()
        test_vector_operations()
        test_batch_operations()
        test_edge_cases()
        test_cosine_similarity()
        test_performance()
        
        print("\n" + "-"*60)
        print("  ALL TESTS PASSED")
        print("-"*60 + "\n")
        
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

main()
