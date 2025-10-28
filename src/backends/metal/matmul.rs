// wtf
use pyo3::prelude::*;
use super::context::MetalContext;
use std::sync::{OnceLock, Mutex};
use crate::python::tensor::Tensor;

static METAL_CTX: OnceLock<Mutex<MetalContext>> = OnceLock::new();

#[pyfunction]
pub fn metal_matmul_f32(a: Tensor, b: Tensor) -> PyResult<Tensor> {
    if a.device != "metal" || b.device != "metal" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Both tensors must be on Metal device"
        ));
    }
    
    let (m, k) = a.shape;
    let (k2, n) = b.shape;
    
    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix dimension mismatch"
        ));
    }

    let ctx = METAL_CTX.get_or_init(|| {
        Mutex::new(MetalContext::new().expect("Failed to initialize Metal context"))
    });

    let mut ctx_guard = ctx.lock().unwrap();
    
    // Get cached output buffer
    let d_c = ctx_guard.buffer_cache.lock().unwrap().get_or_alloc(m, n);
    
    // Compute on Metal GPU (zero copies!)
    ctx_guard.matmul_f32_gpu(a.ptr, b.ptr, d_c, m, n, k, 0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    Ok(Tensor {
        ptr: d_c,
        shape: (m, n),
        device: "metal".to_string(),
        owns_memory: false, // Managed by buffer cache
    })
}
