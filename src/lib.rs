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
    
    // Register NumPy matmul API
    ops::matmul::register(m)?;
    
    // Register GPU tensor matmul with different name to avoid conflict
    #[cfg(target_os = "linux")]
    {
        use pyo3::wrap_pyfunction;
        m.add_function(wrap_pyfunction!(backends::cuda::matmul::matmul, m)?)?;
    }
    
    #[cfg(target_os = "macos")]
    {
        use pyo3::wrap_pyfunction;
        m.add_function(wrap_pyfunction!(backends::metal::matmul::matmul, m)?)?;
    }
    
    Ok(())
}
