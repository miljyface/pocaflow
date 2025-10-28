mod python;
mod core;
mod backends;
mod ops;
mod cpu;

use pyo3::prelude::*;

#[pymodule]
fn pocaflow(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register Tensor class
    python::register(m)?;
    
    // Register smart matmul dispatcher (handles Tensor and NumPy automatically)
    ops::matmul::register(m)?;
    
    Ok(())
}
