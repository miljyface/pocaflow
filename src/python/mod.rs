pub mod array;
pub mod tensor;

use pyo3::prelude::*;

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<tensor::Tensor>()?;
    Ok(())
}
