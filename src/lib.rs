mod python;
mod core;
mod backends;
mod ops;
mod cpu;

use pyo3::prelude::*;

#[pymodule]
fn pocaflow(py: Python, m: &PyModule) -> PyResult<()> {
    python::register(m)?;  // Register Tensor class
    ops::matmul::register(m)?;
    
    #[cfg(target_os = "linux")]
    m.add_function(wrap_pyfunction!(backends::cuda::matmul::matmul, m)?)?;
    
    Ok(())
}