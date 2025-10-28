use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use std::sync::Arc;

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
                let mut ptr: *mut f32 = std::ptr::null_mut();
                let size = m * n * std::mem::size_of::<f32>();
                
                cuda_runtime_sys::cudaMalloc(
                    &mut ptr as *mut _ as *mut *mut std::ffi::c_void,
                    size
                );
                
                cuda_runtime_sys::cudaMemcpy(
                    ptr as *mut std::ffi::c_void,
                    arr.as_ptr() as *const std::ffi::c_void,
                    size,
                    cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyHostToDevice
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
            cuda_runtime_sys::cudaMemcpy(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                m * n * std::mem::size_of::<f32>(),
                cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost
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
        if self.owns_memory && !self.ptr.is_null() {
            unsafe {
                cuda_runtime_sys::cudaFree(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}
