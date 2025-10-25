use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::blas::{dgemm, sgemm};
use super::{pylist_to_array2_f64, pylist_to_array2_f32, array2_to_vec, extract_list};

// parallel batch matrix multiplication for f64
#[pyfunction]
pub fn batch_matmul(
    a: &PyAny,
    b: &PyAny,
    batch_size: usize,
    m: usize,
    k: usize,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let a_list = extract_list(a)?;
    let b_list = extract_list(b)?;
    
    let a_array = pylist_to_array2_f64(a_list)?;
    let b_array = pylist_to_array2_f64(b_list)?;
    
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

#[pyfunction]
pub fn batch_matmul_f32(
    a: &PyAny,
    b: &PyAny,
    batch_size: usize,
    m: usize,
    k: usize,
) -> PyResult<Vec<Vec<Vec<f32>>>> {
    let a_list = extract_list(a)?;
    let b_list = extract_list(b)?;
    
    let a_array = pylist_to_array2_f32(a_list)?;
    let b_array = pylist_to_array2_f32(b_list)?;
    
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

#[pyfunction]
pub fn strided_batch_matmul(
    a: &PyAny,
    b: &PyAny,
    batch_size: usize,
    m: usize,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let a_list = extract_list(a)?;
    let b_list = extract_list(b)?;
    
    let a_array = pylist_to_array2_f64(a_list)?;
    let b_array = pylist_to_array2_f64(b_list)?;
    
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

#[pyfunction]
pub fn strided_batch_matmul_f32(
    a: &PyAny,
    b: &PyAny,
    batch_size: usize,
    m: usize,
) -> PyResult<Vec<Vec<Vec<f32>>>> {
    let a_list = extract_list(a)?;
    let b_list = extract_list(b)?;
    
    let a_array = pylist_to_array2_f32(a_list)?;
    let b_array = pylist_to_array2_f32(b_list)?;
    
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
