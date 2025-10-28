// src/backends/cuda/matmul.rs
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use super::context::CudaContext;
use std::sync::{OnceLock, Mutex};

static CUDA_CTX: OnceLock<Mutex<CudaContext>> = OnceLock::new();

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
        
        // Allocate GPU memory
        let mut d_a = std::ptr::null_mut();
        let mut d_b = std::ptr::null_mut();
        let mut d_c = std::ptr::null_mut();

        unsafe {
            cuda_runtime_sys::cudaMalloc(&mut d_a as *mut _ as *mut *mut std::ffi::c_void, m * k * 4);
            cuda_runtime_sys::cudaMalloc(&mut d_b as *mut _ as *mut *mut std::ffi::c_void, k * n * 4);
            cuda_runtime_sys::cudaMalloc(&mut d_c as *mut _ as *mut *mut std::ffi::c_void, m * n * 4);

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

            // Compute on GPU
            ctx_guard.matmul_f32_gpu(d_a, d_b, d_c, m, n, k, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

            // Copy result back
            let mut result = vec![0.0f32; m * n];
            cuda_runtime_sys::cudaMemcpy(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                d_c as *const std::ffi::c_void,
                m * n * 4,
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost
            );

            cuda_runtime_sys::cudaFree(d_a as *mut std::ffi::c_void);
            cuda_runtime_sys::cudaFree(d_b as *mut std::ffi::c_void);
            cuda_runtime_sys::cudaFree(d_c as *mut std::ffi::c_void);

            ndarray::Array2::from_shape_vec((m, n), result).unwrap()
        }
    };

    Ok(PyArray2::from_owned_array(py, result))
}
