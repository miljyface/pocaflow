use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use super::context::MetalContext;
use std::sync::OnceLock;
use std::sync::Mutex;

static METAL_CTX: OnceLock<Mutex<MetalContext>> = OnceLock::new();

#[pyfunction]
pub fn matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_owned = a.as_array().to_owned();
    let b_owned = b.as_array().to_owned();

    let ctx = METAL_CTX.get_or_init(|| {
        Mutex::new(MetalContext::new().expect("Failed to initialize Metal context"))
    });

    let result = {
        let ctx_guard = ctx.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to lock Metal context: {}", e))
        })?;

        ctx_guard.matmul_f32(&a_owned, &b_owned)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Metal matmul failed: {}", e)))?
    };

    Ok(PyArray2::from_owned_array(py, result))
}