use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
pub fn dot<'py>(
    _py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    Ok(a_arr.iter().zip(b_arr.iter()).map(|(x, y)| x * y).sum())
}

#[pyfunction]
pub fn cross<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();

    let result = vec![
        a_arr[1] * b_arr[2] - a_arr[2] * b_arr[1],
        a_arr[2] * b_arr[0] - a_arr[0] * b_arr[2],
        a_arr[0] * b_arr[1] - a_arr[1] * b_arr[0],
    ];

    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
pub fn magnitude(a: PyReadonlyArray1<'_, f64>) -> f64 {
    a.as_array().iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[pyfunction]
pub fn normalize<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let a_arr = a.as_array();
    let mag: f64 = a_arr.iter().map(|x| x * x).sum::<f64>().sqrt();
    let normalized: Vec<f64> = a_arr.iter().map(|x| x / mag).collect();
    Ok(PyArray1::from_vec(py, normalized))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    Ok(())
}