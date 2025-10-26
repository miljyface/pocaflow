use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use ndarray::s;
use crate::blas::{dgemm, sgemm};

#[pyfunction]
pub fn batch_matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    batch_size: usize,
    m: usize,
    k: usize,
) -> PyResult<Vec<&'py PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let aslice = a.slice(s![i*m..(i+1)*m, ..]);
        let bslice = b.slice(s![i*k..(i+1)*k, ..]);
        let result = dgemm(aslice, bslice);
        results.push(PyArray2::from_owned_array(py, result));
    }
    Ok(results)
}

#[pyfunction]
pub fn batch_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
    batch_size: usize,
    m: usize,
    k: usize,
) -> PyResult<Vec<&'py PyArray2<f32>>> {
    let a = a.as_array();
    let b = b.as_array();

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let aslice = a.slice(s![i*m..(i+1)*m, ..]);
        let bslice = b.slice(s![i*k..(i+1)*k, ..]);
        let result = sgemm(aslice, bslice);
        results.push(PyArray2::from_owned_array(py, result));
    }
    Ok(results)
}
