use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::blas::{dgemm, sgemm};

#[pyfunction]
pub fn batch_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    batch_size: usize,
    m: usize,
    k: usize,
    _n: usize,
) -> PyResult<Vec<&'py PyArray2<f64>>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
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
        .map(|arr| PyArray2::from_owned_array(py, arr))
        .collect())
}

#[pyfunction]
pub fn batch_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f32>,
    b: PyReadonlyArray2<f32>,
    batch_size: usize,
    m: usize,
    k: usize,
    _n: usize,
) -> PyResult<Vec<&'py PyArray2<f32>>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
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
        .map(|arr| PyArray2::from_owned_array(py, arr))
        .collect())
}

#[pyfunction]
pub fn strided_batch_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    batch_size: usize,
    m: usize,
    _n: usize,
) -> PyResult<Vec<&'py PyArray2<f64>>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    let results: Vec<Array2<f64>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a_array.slice(ndarray::s![i*m..(i+1)*m, ..]);
            dgemm(a_slice, b_array)
        })
        .collect();
    
    Ok(results
        .into_iter()
        .map(|arr| PyArray2::from_owned_array(py, arr))
        .collect())
}
