use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};

#[pyfunction]
pub fn dot(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a_vec = a.as_slice()?;
    let b_vec = b.as_slice()?;
    Ok(a_vec.iter().zip(b_vec).map(|(x, y)| x * y).sum())
}

#[pyfunction]
pub fn cross<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray1<f64>> {
    let a_vec = a.as_slice()?;
    let b_vec = b.as_slice()?;
    let result = vec![
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
    ];
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
pub fn magnitude(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a_vec = a.as_slice()?;
    Ok(a_vec.iter().map(|x| x * x).sum::<f64>().sqrt())
}

#[pyfunction]
pub fn normalize<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray1<f64>> {
    let a_vec = a.as_slice()?;
    let mag = a_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    let result: Vec<f64> = a_vec.iter().map(|x| x / mag).collect();
    Ok(PyArray1::from_vec(py, result))
}
