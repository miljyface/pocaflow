use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use super::context::CudaContext;
use std::sync::{OnceLock, Mutex};
use crate::python::tensor::Tensor;

static CUDA_CTX: OnceLock<Mutex<CudaContext>> = OnceLock::new();

// GPU Tensor API (zero-copy)
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
    
    // Get cached output buffer
    let d_c = ctx_guard.buffer_cache.lock().unwrap().get_or_alloc(m, n);
    
    // Compute on GPU (no copies!)
    ctx_guard.matmul_f32_gpu(a.ptr, b.ptr, d_c, m, n, k, 0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    Ok(Tensor {
        ptr: d_c,
        shape: (m, n),
        device: "cuda".to_string(),
        owns_memory: false,
    })
}

// Legacy NumPy API (with copies) - called by ops/matmul.rs
#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let (m, k) = a_arr.dim();
    let (k2, n) = b_arr.dim();

    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Shape mismatch"));
    }

    let ctx = CUDA_CTX.get_or_init(|| {
        Mutex::new(CudaContext::new(4, 256 * 1024 * 1024).expect("CUDA init failed"))
    });

    let result = {
        let mut ctx_guard = ctx.lock().unwrap();
        
        // Get or allocate cached buffers
        let d_a = ctx_guard.buffer_cache.lock().unwrap().get_or_alloc(m, k);
        let d_b = ctx_guard.buffer_cache.lock().unwrap().get_or_alloc(k, n);
        let d_c = ctx_guard.buffer_cache.lock().unwrap().get_or_alloc(m, n);

        unsafe {
            // Copy to GPU
            cuda_runtime_sys::cudaMemcpy(
                d_a as *mut std::ffi::c_void,
                a_arr.as_ptr() as *const std::ffi::c_void,
                m * k * 4,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice
            );
            cuda_runtime_sys::cudaMemcpy(
                d_b as *mut std::ffi::c_void,
                b_arr.as_ptr() as *const std::ffi::c_void,
                k * n * 4,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice
            );

            // Compute
            ctx_guard.matmul_f32_gpu(d_a, d_b, d_c, m, n, k, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

            // Copy back
            let mut result = vec![0.0f32; m * n];
            cuda_runtime_sys::cudaMemcpy(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                d_c as *const std::ffi::c_void,
                m * n * 4,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost
            );

            ndarray::Array2::from_shape_vec((m, n), result).unwrap()
        }
    };

    Ok(PyArray2::from_owned_array(py, result))
}
