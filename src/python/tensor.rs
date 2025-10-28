use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use std::sync::Arc;
use std::ptr;
use std::ffi::c_void;

#[cfg(target_os = "linux")]
use cuda_runtime_sys::{cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyKind, cudaError_t};

#[pyclass]
#[derive(Clone)]
pub struct Tensor {
    pub ptr: *mut f32,
    pub shape: (usize, usize),
    pub device: String,
    pub owns_memory: bool,
}

unsafe impl Send for Tensor {}

#[pymethods]
impl Tensor {
    #[staticmethod]
    pub fn from_array(arr: PyReadonlyArray2<f32>, device: Option<&str>) -> PyResult<Self> {
        let device = device.unwrap_or("cuda");
        let arr = arr.as_array();
        let (m, n) = arr.dim();
        
        if device == "cuda" {
            unsafe {
                let mut ptr: *mut f32 = ptr::null_mut();
                let size = m * n * std::mem::size_of::<f32>();
                
                let ret = cudaMalloc(
                    &mut ptr as *mut _ as *mut *mut c_void,
                    size
                );
                if ret != cudaError_t::cudaSuccess {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("cudaMalloc failed"));
                }
                
                cudaMemcpy(
                    ptr as *mut c_void,
                    arr.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice
                );
                
                Ok(Tensor {
                    ptr,
                    shape: (m, n),
                    device: device.to_string(),
                    owns_memory: true,
                })
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Only 'cuda' device supported"))
        }
    }
    
    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f32>> {
        let (m, n) = self.shape;
        let mut result = vec![0.0f32; m * n];
        
        unsafe {
            cudaMemcpy(
                result.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                m * n * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost
            );
        }
        
        let arr = ndarray::Array2::from_shape_vec((m, n), result)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        Ok(PyArray2::from_owned_array(py, arr))
    }
    
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    pub fn device(&self) -> String {
        self.device.clone()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.owns_memory && !self.ptr.is_null() && self.device == "cuda" {
            #[cfg(target_os = "linux")]
            {
                use crate::backends::cuda::matmul::CUDA_CTX;
                if let Some(ctx) = CUDA_CTX.get() {
                    let (m, n) = self.shape;
                    ctx.lock().unwrap().free(self.ptr, m, n);
                    return;
                }
            }
            
            // Fallback: direct free
            unsafe {
                use std::ffi::c_void;
                use cuda_runtime_sys::cudaFree;
                cudaFree(self.ptr as *mut c_void);
            }
        }
    }
}


