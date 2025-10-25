use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub fn validate_batch_matmul_shape(
    a_shape: (usize, usize),
    b_shape: (usize, usize),
    batch_size: usize,
    m: usize,
    k: usize,
) -> PyResult<()> {
    let (a_rows, a_cols) = a_shape;
    let (b_rows, b_cols) = b_shape;
    if a_rows != batch_size * m {
        return Err(pyo3::exceptions::PyValueError::new_err("A shape mismatch"));
    }
    if b_rows != batch_size * k {
        return Err(pyo3::exceptions::PyValueError::new_err("B shape mismatch"));
    }
    if a_cols != b_cols {
        return Err(pyo3::exceptions::PyValueError::new_err("A and B inner dimensions mismatch"));
    }
    Ok(())
}

pub fn validate_matmul_dims(m: usize, k1: usize, k2: usize, n: usize) -> PyResult<()> {
    if k1 != k2 {
        return Err(PyValueError::new_err(format!(
            "Matrix dimensions don't match: ({}, {}) vs ({}, {})",
            m, k1, k2, n
        )));
    }
    Ok(())
}

// validate that two vectors have the same length
pub fn validate_vector_lengths(len_a: usize, len_b: usize) -> PyResult<()> {
    if len_a != len_b {
        return Err(PyValueError::new_err(format!(
            "Vectors must have same length: {} vs {}",
            len_a, len_b
        )));
    }
    Ok(())
}

// validate that two vectors are both 3D
pub fn validate_3d_vectors(len_a: usize, len_b: usize) -> PyResult<()> {
    if len_a != 3 || len_b != 3 {
        return Err(PyValueError::new_err(
            "Cross product requires 3D vectors (length 3)"
        ));
    }
    Ok(())
}

// validate that a vector is non-zero (for normalization)
pub fn validate_nonzero_magnitude(magnitude: f64) -> PyResult<()> {
    if magnitude == 0.0 {
        return Err(PyValueError::new_err("Cannot normalize zero vector"));
    }
    Ok(())
}
