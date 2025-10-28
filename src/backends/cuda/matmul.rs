use pyo3::prelude::*;
use super::context::CudaContext;
use std::sync::{OnceLock, Arc};
use std::sync::Mutex as StdMutex;
use crate::python::tensor::Tensor;

pub static CUDA_CTX: OnceLock<Arc<StdMutex<CudaContext>>> = OnceLock::new();

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

    // Debug buffer for A
    let debug_a = {
        let mut temp = vec![0.0f32; std::cmp::min(m * k, 8)];
        cudaMemcpy(temp.as_mut_ptr() as *mut c_void, d_a as *const c_void, temp.len() * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        temp
    };
    let debug_b = {
        let mut temp = vec![0.0f32; std::cmp::min(k * n, 8)];
        cudaMemcpy(temp.as_mut_ptr() as *mut c_void, d_b as *const c_void, temp.len() * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        temp
    };
    println!("[HOST] A first 8: {:?}", debug_a);
    println!("[HOST] B first 8: {:?}", debug_b);

    let ctx = CUDA_CTX.get_or_init(|| {
        Arc::new(StdMutex::new(CudaContext::new(4, 256 * 1024 * 1024).expect("CUDA init failed")))
    });

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
