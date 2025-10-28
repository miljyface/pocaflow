use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray2, PyReadonlyArray2};

#[cfg(target_os = "linux")]
use crate::backends::cuda;

#[cfg(target_os = "macos")]
use crate::backends::metal;

use crate::cpu::blas::{sgemm, dgemm};

#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    #[cfg(target_os = "linux")]
    return cuda::matmul::cuda_matmul_f32(py, a, b);
    
    #[cfg(target_os = "macos")]
    return metal::matmul::metal_matmul_f32(py, a, b);
    
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return matmul_f32_cpu(py, a, b);
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

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f32_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f64, m)?)?;
    Ok(())
}