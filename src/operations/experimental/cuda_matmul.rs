use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use crate::gpu::CudaContext;
use std::sync::OnceLock;
use std::sync::Mutex;

// Global context with OnceLock for better performance
static CUDA_CTX: OnceLock<Mutex<CudaContext>> = OnceLock::new();

#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: &PyAny,
    b: &PyAny,
) -> PyResult<PyObject> {
    // Check if inputs are already CUDA tensors
    let a_device = a.getattr("device")?.str()?.to_string();
    let b_device = b.getattr("device")?.str()?.to_string();

    if a_device.contains("cuda") && b_device.contains("cuda") {
        // FAST PATH: Already on GPU, use PyTorch's matmul
        let torch = py.import("torch")?;
        let result = torch.call_method1("matmul", (a, b))?;
        return Ok(result.into());
    }

    // SLOW PATH: CPU tensors, use our CUDA backend
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

    // Initialize or get global context
    let ctx = CUDA_CTX.get_or_init(|| {
        // Pre-allocate for 4096x4096 matrices
        let initial_size = 4096 * 4096;
        Mutex::new(
            CudaContext::new(initial_size, initial_size, initial_size)
                .expect("Failed to initialize CUDA context")
        )
    });

    // Perform matrix multiplication
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