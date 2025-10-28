use pyo3::prelude::*;
use pyo3::exceptions;
use numpy::{PyArray2, Element};
use std::sync::Arc;
use std::ptr;
use std::ffi::c_void;
use ndarray::ShapeBuilder;

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
    pub fn from_array(_py: Python, arr: &PyAny, device: usize) -> PyResult<Self> {
        let arr = arr.downcast::<PyArray2<f32>>()
            .map_err(|_| PyErr::new::<exceptions::PyTypeError, _>(
                "Input must be a 2D numpy.ndarray of float32"))?;

        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyErr::new::<exceptions::PyValueError, _>(
                "Only 2D arrays supported for GPU matmul"));
        }
        // Enforce Fortran order for cuBLAS compatibility
        if !arr.is_fortran_contiguous() {
            return Err(PyErr::new::<exceptions::PyValueError, _>(
                "Array must be Fortran-contiguous (use np.asfortranarray)"));
        }
        let (m, n) = (shape[0], shape[1]);
        let numel = m * n;
        let size = numel * std::mem::size_of::<f32>();

        // Allocate CUDA memory
        let mut d_ptr: *mut f32 = ptr::null_mut();
        unsafe {
            cudaMalloc(&mut d_ptr as *mut _ as *mut *mut c_void, size);
            // Copy Fortran-order memory block
            cudaMemcpy(
                d_ptr as *mut c_void,
                arr.data() as *const c_void,
                size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
        }

        Ok(Tensor {
            ptr: d_ptr,
            shape: (m, n),
            device: "cuda".to_string(),
            owns_memory: true,
        })
    }

    pub fn to_array(&self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
        let (m, n) = self.shape;
        unsafe {
            let host_buf = PyArray2::<f32>::new(py, [m as usize, n as usize], false); // F-order!
            let dest_ptr = host_buf.as_ptr();
            // Device to host copy
            cudaMemcpy(
                dest_ptr as *mut c_void,
                self.ptr as *const c_void,
                (m * n) * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            Ok(host_buf.to_owned())
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
        
        // CRITICAL: Fortran order!
        let arr = ndarray::Array2::from_shape_vec((m, n).f(), result)
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
            unsafe {
                cudaFree(self.ptr as *mut c_void);
            }
        }
    }
}
