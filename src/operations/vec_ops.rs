use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::utils::{
    validate_vector_lengths, 
    validate_3d_vectors, 
    validate_nonzero_magnitude,
    validate_nonzero_magnitude_f32
};

/// Helper to extract list data from Tensor or direct list
fn extract_vec_f64(obj: &PyAny) -> PyResult<Vec<f64>> {
    // Try to get .data attribute first (for Tensor objects)
    if let Ok(data) = obj.getattr("data") {
        data.extract::<&PyList>()?.extract()
    } else {
        // Fallback to direct extraction (for plain lists)
        obj.extract::<&PyList>()?.extract()
    }
}

/// Helper to extract list data from Tensor or direct list (f32)
fn extract_vec_f32(obj: &PyAny) -> PyResult<Vec<f32>> {
    if let Ok(data) = obj.getattr("data") {
        data.extract::<&PyList>()?.extract()
    } else {
        obj.extract::<&PyList>()?.extract()
    }
}

#[pyfunction]
pub fn dot(a: &PyAny, b: &PyAny) -> PyResult<f64> {
    let a_vec = extract_vec_f64(a)?;
    let b_vec = extract_vec_f64(b)?;
    
    validate_vector_lengths(a_vec.len(), b_vec.len())?;
    
    let result: f64 = a_vec.iter()
        .zip(b_vec.iter())
        .map(|(x, y)| x * y)
        .sum();
    
    Ok(result)
}

#[pyfunction]
pub fn dot_f32(a: &PyAny, b: &PyAny) -> PyResult<f32> {
    let a_vec = extract_vec_f32(a)?;
    let b_vec = extract_vec_f32(b)?;
    
    validate_vector_lengths(a_vec.len(), b_vec.len())?;
    
    let result: f32 = a_vec.iter()
        .zip(b_vec.iter())
        .map(|(x, y)| x * y)
        .sum();
    
    Ok(result)
}

#[pyfunction]
pub fn cross(a: &PyAny, b: &PyAny) -> PyResult<Vec<f64>> {
    let a_vec = extract_vec_f64(a)?;
    let b_vec = extract_vec_f64(b)?;
    
    validate_3d_vectors(a_vec.len(), b_vec.len())?;
    
    let result = vec![
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
    ];
    
    Ok(result)
}

#[pyfunction]
pub fn cross_f32(a: &PyAny, b: &PyAny) -> PyResult<Vec<f32>> {
    let a_vec = extract_vec_f32(a)?;
    let b_vec = extract_vec_f32(b)?;
    
    validate_3d_vectors(a_vec.len(), b_vec.len())?;
    
    let result = vec![
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
    ];
    
    Ok(result)
}

#[pyfunction]
pub fn magnitude(a: &PyAny) -> PyResult<f64> {
    let a_vec = extract_vec_f64(a)?;
    let result = a_vec.iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    Ok(result)
}

#[pyfunction]
pub fn magnitude_f32(a: &PyAny) -> PyResult<f32> {
    let a_vec = extract_vec_f32(a)?;
    let result = a_vec.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    Ok(result)
}

#[pyfunction]
pub fn normalize(a: &PyAny) -> PyResult<Vec<f64>> {
    let a_vec = extract_vec_f64(a)?;
    let mag = a_vec.iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    validate_nonzero_magnitude(mag)?;
    let result: Vec<f64> = a_vec.iter()
        .map(|x| x / mag)
        .collect();
    Ok(result)
}

#[pyfunction]
pub fn normalize_f32(a: &PyAny) -> PyResult<Vec<f32>> {
    let a_vec = extract_vec_f32(a)?;
    let mag = a_vec.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    validate_nonzero_magnitude_f32(mag)?;
    let result: Vec<f32> = a_vec.iter()
        .map(|x| x / mag)
        .collect();
    Ok(result)
}
