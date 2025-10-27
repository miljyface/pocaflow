use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::gpu::CudaContext;

#[pyfunction]
pub fn cudamatmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>
) -> PyResult<&'py PyArray2<f32>> {
    let ctx = CudaContext::new();
    let a_owned = a.as_array().to_owned();
    let b_owned = b.as_array().to_owned();
    let result = ctx.matmul_f32(&a_owned, &b_owned);
    Ok(PyArray2::from_owned_array(py, result))
}

