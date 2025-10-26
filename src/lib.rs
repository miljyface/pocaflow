use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod blas;
mod operations;
mod utils;

#[pymodule]
fn rust_linalg(_py: Python, m: &PyModule) -> PyResult<()> {

    // Matrix operations
    m.add_function(wrap_pyfunction!(operations::matmul::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(operations::matmul::matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(operations::matmul::matmul_f64, m)?)?;
    m.add_function(wrap_pyfunction!(operations::batch::batch_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(operations::batch::batch_matmul_f64, m)?)?;

    // Strassen Algo
    m.add_function(wrap_pyfunction!(operations::experimental::strassen::strassen_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(operations::experimental::strassen::strassen_matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(operations::experimental::strassen::strassen_matmul_f64, m)?)?;
    
    // Vector operations
    m.add_function(wrap_pyfunction!(operations::vec_ops::dot, m)?)?;
    m.add_function(wrap_pyfunction!(operations::vec_ops::cross, m)?)?;
    m.add_function(wrap_pyfunction!(operations::vec_ops::magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(operations::vec_ops::normalize, m)?)?;
    
    Ok(())
}

