use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::gpu::CudaContext;
use std::sync::OnceLock;

static CUDA_CTX: OnceLock<CudaContext> = OnceLock::new();

#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();

    let (m, k1) = a_arr.dim();
    let (k2, n) = b_arr.dim();

    if k1 != k2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
    }

    let ctx = CUDA_CTX.get_or_init(|| CudaContext::new());
    let a_owned = a_arr.to_owned();
    let b_owned = b_arr.to_owned();

    let result = py.allow_threads(|| {
        ctx.matmul_f32(&a_owned, &b_owned)
    });

    Ok(PyArray2::from_owned_array(py, result))
}
