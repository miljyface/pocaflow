use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::blas::{CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, cblas_dgemm, cblas_sgemm};

#[cfg(target_os = "macos")]
use super::experimental::metal_matmul::metal_matmul_f32 as gpu_matmul_f32;

#[cfg(target_os = "linux")]
use super::experimental::metal_matmul::cuda_matmul_f32 as gpu_matmul_f32;

#[cfg(target_os = "windows")]
use super::experimental::metal_matmul::cuda_matmul_f32 as gpu_matmul_f32;


// default matmul is gpu f32 
#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
        gpu_matmul_f32(py, a, b)
}

// force use cpu
#[pyfunction]
pub fn matmul_f32_cpu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let (m, k1) = a_arr.dim();
    let (k2, n) = b_arr.dim();
    
    debug_assert_eq!(k1, k2);
    
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    let a_owned = a_arr.to_owned();
    let b_owned = b_arr.to_owned();
    let mut result = ndarray::Array2::<f32>::zeros((m, n));

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k1 as i32,
            1.0,
            a_owned.as_ptr(), k1 as i32,
            b_owned.as_ptr(), n as i32,
            0.0,
            result.as_mut_ptr(), n as i32,
        );
    }

    Ok(PyArray2::from_owned_array(py, result))
}

// metal cant do f64 so f64 is cpu only (consistent with ptorch)
#[pyfunction]
pub fn matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let (m, k1) = a_arr.dim();
    let (k2, n) = b_arr.dim();
    
    debug_assert_eq!(k1, k2);
    
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    let a_owned = a_arr.to_owned();
    let b_owned = b_arr.to_owned();
    let mut result = ndarray::Array2::<f64>::zeros((m, n));

    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k1 as i32,
            1.0,
            a_owned.as_ptr(), k1 as i32,
            b_owned.as_ptr(), n as i32,
            0.0,
            result.as_mut_ptr(), n as i32,
        );
    }

    Ok(PyArray2::from_owned_array(py, result))
}
