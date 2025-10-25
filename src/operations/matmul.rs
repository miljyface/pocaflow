use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::utils::validate_matmul_dims;
use crate::blas::dgemm;

#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>
) -> PyResult<&'py PyArray2<f64>> {
    let a = a.as_array();
    let b = b.as_array();

    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    validate_matmul_dims(m, k1, k2, n)?;

    let result = dgemm(
        a.view().into_dimensionality::<ndarray::Ix2>().unwrap(),
        b.view().into_dimensionality::<ndarray::Ix2>().unwrap()
    );

    Ok(PyArray2::from_array(py, &result))
}
