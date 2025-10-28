use pyo3::prelude::*;

mod core;
mod backends;
mod cpu;
mod ops;
mod python;
mod utils;

#[pymodule]
fn pocaflow(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register operations
    ops::register(m)?;
    m.add_class::<python::tensor::Tensor>()?;
    Ok(())
}