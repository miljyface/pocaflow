use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod blas;
mod operations;
mod utils;

mod gpu;

#[pymodule]
fn pocaflow(_py: Python, m: &PyModule) -> PyResult<()> {
    
    m.add_function(wrap_pyfunction!(operations::matmul::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(operations::matmul::matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(operations::matmul::matmul_f32_cpu, m)?)?;

    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(
        operations::experimental::metal_matmul::metal_matmul_f32,
        m
    )?)?;

    #[cfg(not(target_os = "macos"))]
    m.add_function(wrap_pyfunction!(
        operations::experimental::cuda_matmul::cuda_matmul_f32,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(operations::vec_ops::dot, m)?)?;
    m.add_function(wrap_pyfunction!(operations::vec_ops::cross, m)?)?;
    m.add_function(wrap_pyfunction!(operations::vec_ops::magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(operations::vec_ops::normalize, m)?)?;

    Ok(())
}
