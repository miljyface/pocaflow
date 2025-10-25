use pyo3::prelude::*;
use pyo3::types::PyList;
use ndarray::Array2;
use crate::blas::{dgemm, sgemm};
use crate::utils::validate_matmul_dims;
use super::{pylist_to_array2_f64, pylist_to_array2_f32, array2_to_vec};

fn extract_list(obj: &PyAny) -> PyResult<&PyList> {
    // Try to get .data attribute first (for Tensor objects)
    if let Ok(data) = obj.getattr("data") {
        data.extract()
    } else {
        // Fallback to direct extraction (for plain lists)
        obj.extract()
    }
}

pub fn matmul_impl(a: &PyAny, b: &PyAny) -> PyResult<Array2<f64>> {
    let a_list = extract_list(a)?;
    let b_list = extract_list(b)?;
    
    let a_array = pylist_to_array2_f64(a_list)?;
    let b_array = pylist_to_array2_f64(b_list)?;
    
    let (m, k1) = (a_array.nrows(), a_array.ncols());
    let (k2, n) = (b_array.nrows(), b_array.ncols());
    
    validate_matmul_dims(m, k1, k2, n)?;
    
    Ok(dgemm(a_array.view(), b_array.view()))
}

/// Internal implementation for f32 matrix multiplication
pub fn matmul_f32_impl(a: &PyAny, b: &PyAny) -> PyResult<Array2<f32>> {
    let a_list = extract_list(a)?;
    let b_list = extract_list(b)?;
    
    let a_array = pylist_to_array2_f32(a_list)?;
    let b_array = pylist_to_array2_f32(b_list)?;
    
    let (m, k1) = (a_array.nrows(), a_array.ncols());
    let (k2, n) = (b_array.nrows(), b_array.ncols());
    
    validate_matmul_dims(m, k1, k2, n)?;
    
    Ok(sgemm(a_array.view(), b_array.view()))
}

/// Matrix multiplication (f64) - Python interface
#[pyfunction]
pub fn matmul(a: &PyAny, b: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    let result = matmul_impl(a, b)?;
    Ok(array2_to_vec(result))
}

/// Matrix multiplication (f32) - Python interface
#[pyfunction]
pub fn matmul_f32(a: &PyAny, b: &PyAny) -> PyResult<Vec<Vec<f32>>> {
    let result = matmul_f32_impl(a, b)?;
    Ok(array2_to_vec(result))
}
