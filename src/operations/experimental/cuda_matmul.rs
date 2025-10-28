use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use crate::gpu::CudaContext;
use std::sync::OnceLock;
use std::sync::Mutex;

// Safe static context for Rust
static CUDA_CTX: OnceLock<Mutex<CudaContext>> = OnceLock::new();

#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: &PyAny,
    b: &PyAny,
) -> PyResult<PyObject> {
    // Check for PyTorch CUDA tensors (fast path)
    let a_device = a.getattr("device").and_then(|dev| dev.str()).unwrap_or("<cpu>").to_string();
    let b_device = b.getattr("device").and_then(|dev| dev.str()).unwrap_or("<cpu>").to_string();

    if a_device.contains("cuda") && b_device.contains("cuda") {
        // Use PyTorch native matmul for CUDA tensors
        let torch = py.import("torch")?;
        let result = torch.call_method1("matmul", (a, b))?;
        return Ok(result.into());
    }

    // Slow path: expects numpy arrays/tensors convertible to f32 array
    let a_array = a.extract::<PyReadonlyArray2<f32>>()?;
    let b_array = b.extract::<PyReadonlyArray2<f32>>()?;

    let a_owned = a_array.as_array().to_owned();
    let b_owned = b_array.as_array().to_owned();

    let m = a_owned.shape()[0];
    let k = a_owned.shape()[1];
    let n = b_owned.shape()[1];
    let max_a_elems = m * k;
    let max_b_elems = k * n;
    let max_c_elems = m * n;

    let ctx = CUDA_CTX.get_or_init(|| {
        let initial_size = 4096 * 4096;
        Mutex::new(
            CudaContext::new(initial_size, initial_size, initial_size)
                .expect("Failed to initialize CUDA context")
        )
    });

    let result = {
        let mut ctx_guard = ctx.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to lock CUDA context: {}", e)
            )
        })?;

        ctx_guard.ensure_capacity(max_a_elems, max_b_elems, max_c_elems)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("CUDA ensure_capacity failed: {}", e)
                )
            })?;

        ctx_guard.matmul_f32(&a_owned, &b_owned)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("CUDA matmul failed: {}", e)
                )
            })?
    };

    Ok(PyArray2::from_owned_array(py, result).into())
}
