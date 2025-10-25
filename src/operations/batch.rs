use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use ndarray::s;
use crate::blas::dgemm;
use crate::utils::validate_batch_matmul_shape;

#[pyfunction]
pub fn batch_matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    batch_size: usize,
    m: usize,
    k: usize,
) -> PyResult<Vec<&'py PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();

    validate_batch_matmul_shape(a.dim(), b.dim(), batch_size, m, k)?;

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let aslice = a.slice(s![i*m..(i+1)*m, ..]);
        let bslice = b.slice(s![i*k..(i+1)*k, ..]);
        let result = dgemm(aslice, bslice);
        results.push(PyArray2::<f64>::from_array(py, &result));
    }

    Ok(results)
}
