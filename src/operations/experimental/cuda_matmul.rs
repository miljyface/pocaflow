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
    // We'll propagate errors, converting your custom i32 errors to PyErr
    let result = CUDA_CTX.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        if ctx_ref.is_none() {
            // Unwrap or map error if context creation fails:
            *ctx_ref = Some(CudaContext::new().map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to init CUDA context: {}", e))
            )?);
        }
        // matmul_f32 returns a Result as well, propagate it:
        ctx_ref.as_ref().unwrap().matmul_f32(&a_owned, &b_owned)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CUDA matmul failed: {}", e)))
    })?;
    // At this point, 'result' is Array2<f32>, so pass it to numpy
    Ok(PyArray2::from_owned_array(py, result))
}
