use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use crate::gpu::CudaContext;
use std::sync::{OnceLock, Mutex};

static CUDA_CTX: OnceLock<Mutex<CudaContext>> = OnceLock::new();

#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_owned = a.as_array().to_owned();
    let b_owned = b.as_array().to_owned();
    let m = a_owned.shape()[0];
    let k = a_owned.shape()[1];
    let n = b_owned.shape()[1];
    let max_a_elems = m * k;
    let max_b_elems = k * n;
    let max_c_elems = m * n;
    let n_streams = 4;
    let workspace_size = 128 * 1024 * 1024;

    let ctx = CUDA_CTX.get_or_init(|| {
        let initial_size = 4096 * 4096;
        Mutex::new(CudaContext::new(
            initial_size, initial_size, initial_size, n_streams, workspace_size)
            .expect("Failed to initialize CUDA context"))
    });

    let result = {
        let mut ctx_guard = ctx.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to lock CUDA context: {}", e))
        })?;

        ctx_guard.ensure_capacity(max_a_elems, max_b_elems, max_c_elems)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("CUDA ensure_capacity failed: {}", e)))?;

        // For this PyO3 function we're using only stream 0
        ctx_guard.matmul_f32(&a_owned, &b_owned, 0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("CUDA matmul failed: {}", e)))?
    };

    Ok(PyArray2::from_owned_array(py, result))
}