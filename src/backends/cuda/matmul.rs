use pyo3::prelude::*;
use super::context::CudaContext;
use std::sync::{OnceLock, Mutex};
use crate::python::tensor::Tensor;

static CUDA_CTX: OnceLock<Mutex<CudaContext>> = OnceLock::new();

#[pyfunction]
pub fn matmul(a: Tensor, b: Tensor) -> PyResult<Tensor> {
    if a.device != "cuda" || b.device != "cuda" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Both tensors must be on CUDA"
        ));
    }
    
    let (m, k) = a.shape;
    let (k2, n) = b.shape;
    
    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Shape mismatch"));
    }

    let ctx = CUDA_CTX.get_or_init(|| {
        Mutex::new(CudaContext::new(4, 256 * 1024 * 1024).expect("CUDA init failed"))
    });

    let mut ctx_guard = ctx.lock().unwrap();
    
    // Get cached output buffer (reuse across calls)
    let d_c = ctx_guard.buffer_cache.lock().unwrap().get_or_alloc(m, n);
    
    // Compute on GPU
    ctx_guard.matmul_f32_gpu(a.ptr, b.ptr, d_c, m, n, k, 0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    Ok(Tensor {
        ptr: d_c,
        shape: (m, n),
        device: "cuda".to_string(),
        owns_memory: false, // IMPORTANT: Cache owns it, don't free on drop
    })
}
