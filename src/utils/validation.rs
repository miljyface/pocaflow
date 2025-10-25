use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub fn validate_matmul_dims(m: usize, k1: usize, k2: usize, n: usize) -> PyResult<()> {
    if k1 != k2 {
        return Err(PyValueError::new_err(format!(
            "Matrix dimensions don't match: ({}, {}) vs ({}, {})",
            m, k1, k2, n
        )));
    }
    Ok(())
}
