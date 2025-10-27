use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::gpu::CudaContext;
use std::cell::RefCell;

thread_local! {
    static CUDA_CTX: RefCell<Option<CudaContext>> = RefCell::new(None);
}

#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_owned = a.as_array().to_owned();
    let b_owned = b.as_array().to_owned();
    let result = CUDA_CTX.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        if ctx_ref.is_none() {
            *ctx_ref = Some(CudaContext::new());
        }
        ctx_ref.as_ref().unwrap().matmul_f32(&a_owned, &b_owned)
    });
    Ok(PyArray2::from_owned_array(py, result))
}
