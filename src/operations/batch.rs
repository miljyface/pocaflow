use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use crate::blas::{dgemm, sgemm};
use super::{pylist_to_array2_f64, pylist_to_array2_f32, array2_to_vec};

/// Batch matrix multiplication (parallel)
#[pyfunction]
pub fn batch_matmul(
    a: &PyList,
    b: &PyList,
    batch_size: usize,
    m: usize,
    k: usize,
    _n: usize,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let a_array = pylist_to_array2_f64(a)?;
    let b_array = pylist_to_array2_f64(b)?;
    
    // Validate dimensions
    if a_array.nrows() != batch_size * m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Matrix A rows mismatch: expected {}, got {}",
            batch_size * m,
            a_array.nrows()
        )));
    }
    
    let results: Vec<Array2<f64>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a_array.slice(ndarray::s![i*m..(i+1)*m, ..]);
            let b_slice = b_array.slice(ndarray::s![i*k..(i+1)*k, ..]);
            dgemm(a_slice, b_slice)
        })
        .collect();
    
    Ok(results
        .into_iter()
        .map(array2_to_vec)
        .collect())
}

/// Batch matrix multiplication (f32, parallel)
#[pyfunction]
pub fn batch_matmul_f32(
    a: &PyList,
    b: &PyList,
    batch_size: usize,
    m: usize,
    k: usize,
    _n: usize,
) -> PyResult<Vec<Vec<Vec<f32>>>> {
    let a_array = pylist_to_array2_f32(a)?;
    let b_array = pylist_to_array2_f32(b)?;
    
    if a_array.nrows() != batch_size * m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Matrix A rows mismatch: expected {}, got {}",
            batch_size * m,
            a_array.nrows()
        )));
    }
    
    let results: Vec<Array2<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a_array.slice(ndarray::s![i*m..(i+1)*m, ..]);
            let b_slice = b_array.slice(ndarray::s![i*k..(i+1)*k, ..]);
            sgemm(a_slice, b_slice)
        })
        .collect();
    
    Ok(results
        .into_iter()
        .map(array2_to_vec)
        .collect())
}

/// Strided batch matrix multiplication (shared B matrix, parallel)
#[pyfunction]
pub fn strided_batch_matmul(
    a: &PyList,
    b: &PyList,
    batch_size: usize,
    m: usize,
    _k: usize,
    _n: usize,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let a_array = pylist_to_array2_f64(a)?;
    let b_array = pylist_to_array2_f64(b)?;
    
    if a_array.nrows() != batch_size * m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Matrix A rows mismatch: expected {}, got {}",
            batch_size * m,
            a_array.nrows()
        )));
    }
    
    let results: Vec<Array2<f64>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a_array.slice(ndarray::s![i*m..(i+1)*m, ..]);
            dgemm(a_slice, b_array.view())
        })
        .collect();
    
    Ok(results
        .into_iter()
        .map(array2_to_vec)
        .collect())
}

/// Strided batch matrix multiplication (f32, shared B matrix, parallel)
#[pyfunction]
pub fn strided_batch_matmul_f32(
    a: &PyList,
    b: &PyList,
    batch_size: usize,
    m: usize,
    _k: usize,
    _n: usize,
) -> PyResult<Vec<Vec<Vec<f32>>>> {
    let a_array = pylist_to_array2_f32(a)?;
    let b_array = pylist_to_array2_f32(b)?;
    
    if a_array.nrows() != batch_size * m {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Matrix A rows mismatch: expected {}, got {}",
            batch_size * m,
            a_array.nrows()
        )));
    }
    
    let results: Vec<Array2<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a_array.slice(ndarray::s![i*m..(i+1)*m, ..]);
            sgemm(a_slice, b_array.view())
        })
        .collect();
    
    Ok(results
        .into_iter()
        .map(array2_to_vec)
        .collect())
}
