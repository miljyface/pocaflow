use pyo3::prelude::*;
use ndarray::Array2;
use crate::blas::{dgemm, sgemm};
use crate::utils::validate_matmul_dims;
use super::{pylist_to_array2_f64, pylist_to_array2_f32, array2_to_vec, extract_list};

// implicated matmul functions don't call these LOL
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

/// call these one in python
#[pyfunction]
pub fn matmul(a: &PyAny, b: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    let result = matmul_impl(a, b)?;
    Ok(array2_to_vec(result))
}

#[pyfunction]
pub fn matmul_f32(a: &PyAny, b: &PyAny) -> PyResult<Vec<Vec<f32>>> {
    let result = matmul_f32_impl(a, b)?;
    Ok(array2_to_vec(result))
}
