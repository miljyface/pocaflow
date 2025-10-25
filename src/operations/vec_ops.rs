use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::validate_vector_lengths;
use crate::utils::validate_3d_vectors;
use crate::utils::validate_nonzero_magnitude;
use crate::utils::validate_nonzero_magnitude_f32;

/// Vector dot product (inner product)
/// Returns a scalar: sum(a[i] * b[i])
#[pyfunction]
pub fn dot(
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    validate_vector_lengths(a_array.len(), b_array.len())?;
    
    let result: f64 = a_array.iter()
        .zip(b_array.iter())
        .map(|(x, y)| x * y)
        .sum();
    
    Ok(result)
}

/// Vector dot product for f32
#[pyfunction]
pub fn dot_f32(
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    validate_vector_lengths(a_array.len(), b_array.len())?;
    
    let result: f32 = a_array.iter()
        .zip(b_array.iter())
        .map(|(x, y)| x * y)
        .sum();
    
    Ok(result)
}

/// Vector cross product (only for 3D vectors)
/// Returns a 3D vector perpendicular to both inputs
#[pyfunction]
pub fn cross<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    validate_3d_vectors(a_array.len(), b_array.len())?;
    
    let a0 = a_array[0];
    let a1 = a_array[1];
    let a2 = a_array[2];
    
    let b0 = b_array[0];
    let b1 = b_array[1];
    let b2 = b_array[2];
    
    // Cross product formula: a × b = (a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0)
    let result = vec![
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ];
    
    Ok(PyArray1::from_vec(py, result))
}

/// Vector cross product for f32
#[pyfunction]
pub fn cross_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let a_array = a.as_array();
    let b_array = b.as_array();
    
    validate_3d_vectors(a_array.len(), b_array.len())?;
    
    let a0 = a_array[0];
    let a1 = a_array[1];
    let a2 = a_array[2];
    
    let b0 = b_array[0];
    let b1 = b_array[1];
    let b2 = b_array[2];
    
    let result = vec![
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ];
    
    Ok(PyArray1::from_vec(py, result))
}

/// Vector magnitude (L2 norm)
#[pyfunction]
pub fn magnitude(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a_array = a.as_array();
    let result = a_array.iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    Ok(result)
}

/// Vector magnitude for f32
#[pyfunction]
pub fn magnitude_f32(a: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let a_array = a.as_array();
    let result = a_array.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    Ok(result)
}

/// Vector normalization (unit vector)
#[pyfunction]
pub fn normalize<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let a_array = a.as_array();
    
    let mag = a_array.iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    
    validate_nonzero_magnitude(mag)?;
    
    let result: Vec<f64> = a_array.iter()
        .map(|x| x / mag)
        .collect();
    
    Ok(PyArray1::from_vec(py, result))
}

/// Vector normalization for f32
#[pyfunction]
pub fn normalize_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f32>,
) -> PyResult<&'py PyArray1<f32>> {
    let a_array = a.as_array();
    
    let mag = a_array.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    
    validate_nonzero_magnitude_f32(mag)?;
    
    let result: Vec<f32> = a_array.iter()
        .map(|x| x / mag)
        .collect();
    
    Ok(PyArray1::from_vec(py, result))
}
