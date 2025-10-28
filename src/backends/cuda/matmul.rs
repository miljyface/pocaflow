use pyo3::prelude::*;
use super::context::CudaContext;
use std::sync::{OnceLock, Arc};
use std::sync::Mutex as StdMutex;
use crate::python::tensor::Tensor;
use std::ptr;

static CUDA_CTX: OnceLock<Arc<StdMutex<CudaContext>>> = OnceLock::new();

#[pyfunction]
pub fn matmul(a: Tensor, b: Tensor) -> PyResult<Tensor> {
    if a.device != "cuda" || b.device != "cuda" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Both tensors must be on CUDA"));
    }
    
    let (m, k) = a.shape;
    let (k2, n) = b.shape;
    
    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Shape mismatch"));
    }

    let ctx = CUDA_CTX.get_or_init(|| {
        Arc::new(StdMutex::new(CudaContext::new(4, 256 * 1024 * 1024).expect("CUDA init failed")))
    });

    // Get buffer from pool
    let d_c = ctx.lock().unwrap().alloc(m, n);
    
    ctx.lock().unwrap().matmul_f32_gpu(a.ptr, b.ptr, d_c, m, n, k, 0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    Ok(Tensor {
        ptr: d_c,
        shape: (m, n),
        device: "cuda".to_string(),
        owns_memory: true,
    })
}
