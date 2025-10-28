use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::blas::{dgemm, sgemm};

#[cfg(target_os = "macos")]
use crate::operations::experimental::metal_matmul::metal_matmul_f32 as gpu_matmul_f32;

#[cfg(target_os = "linux")]
use crate::operations::experimental::cuda_matmul::cuda_matmul_f32 as gpu_matmul_f32;

// GPU matmul entry (returns a PyArray2<f32> directly)
#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    gpu_matmul_f32(py, a, b)
}

// CPU fallback, returns a reference to PyArray2<f32>
#[pyfunction]
pub fn matmul_f32_cpu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    Ok(PyArray2::from_owned_array(py, sgemm(a.as_array(), b.as_array())))
}

// CPU double precision fallback
#[pyfunction]
pub fn matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    Ok(PyArray2::from_owned_array(py, dgemm(a.as_array(), b.as_array())))
}
