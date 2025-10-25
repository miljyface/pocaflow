use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::blas::{dgemm, sgemm};
use crate::utils::validate_matmul_dims;

pub fn matmul_impl(
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
) -> PyResult<Array2<f64>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    let (m, k1) = (a_array.nrows(), a_array.ncols());
    let (k2, n) = (b_array.nrows(), b_array.ncols());
    
    validate_matmul_dims(m, k1, k2, n)?;
    
    Ok(dgemm(a_array, b_array))
}

pub fn matmul_f32_impl(
    a: PyReadonlyArray2<f32>,
    b: PyReadonlyArray2<f32>,
) -> PyResult<Array2<f32>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    let (m, k1) = (a_array.nrows(), a_array.ncols());
    let (k2, n) = (b_array.nrows(), b_array.ncols());
    
    validate_matmul_dims(m, k1, k2, n)?;
    
    Ok(sgemm(a_array, b_array))
}

#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let result = matmul_impl(a, b)?;
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
pub fn matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f32>,
    b: PyReadonlyArray2<f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let result = matmul_f32_impl(a, b)?;
    Ok(PyArray2::from_owned_array(py, result))
}
