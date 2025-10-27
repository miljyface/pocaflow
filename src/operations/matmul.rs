use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::blas::{dgemm, sgemm};

#[cfg(target_os = "macos")]
use super::experimental::metal_matmul::metal_matmul_f32 as gpu_matmul_f32;

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod gpu_stub {
    use numpy::{PyReadonlyArray2, PyArray2};
    use pyo3::prelude::*;
    pub fn gpu_matmul_f32<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f32>,
        b: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        // Fallback to CPU implementation
        Ok(PyArray2::from_owned_array(py, a.as_array().dot(&b.as_array())))
    }
}
#[cfg(any(target_os = "linux", target_os = "windows"))]
use gpu_stub::gpu_matmul_f32;

#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    gpu_matmul_f32(py, a, b)
}

#[pyfunction]
pub fn matmul_f32_cpu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    Ok(PyArray2::from_owned_array(py, sgemm(a.as_array(), b.as_array())))
}

#[pyfunction]
pub fn matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    Ok(PyArray2::from_owned_array(py, dgemm(a.as_array(), b.as_array())))
}
