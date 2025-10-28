use pyo3::prelude::*;

pub mod matmul;
pub mod vec_ops;

pub fn register(m: &PyModule) -> PyResult<()> {
    matmul::register(m)?;
    vec_ops::register(m)?;
    Ok(())
}