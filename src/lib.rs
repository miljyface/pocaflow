use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod blas;
mod operations;
mod utils;

#[pymodule]
fn rust_linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(operations::matmul::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(operations::matmul::matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(operations::batch::batch_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(operations::batch::batch_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(operations::batch::strided_batch_matmul, m)?)?;
    Ok(())
}
