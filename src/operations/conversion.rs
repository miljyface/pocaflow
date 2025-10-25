use pyo3::prelude::*;
use pyo3::types::PyList;
use ndarray::Array2;

pub fn extract_list(obj: &PyAny) -> PyResult<&PyList> {
    // try to get .data attribute first from tensors
    if let Ok(data) = obj.getattr("data") {
        data.extract()
    } else {
        // fallback
        obj.extract()
    }
}

// convert nested list structure to f64 array2
pub fn pylist_to_array2_f64(list: &PyList) -> PyResult<Array2<f64>> {
    let mut rows = Vec::new();
    let mut ncols = 0;
    
    for item in list.iter() {
        let row: Vec<f64> = item.extract::<&PyList>()?.extract()?;
        if ncols == 0 {
            ncols = row.len();
        } else if row.len() != ncols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All rows must have same length"
            ));
        }
        rows.extend(row);
    }
    
    let nrows = list.len();
    Array2::from_shape_vec((nrows, ncols), rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// same thing for f32 array2
pub fn pylist_to_array2_f32(list: &PyList) -> PyResult<Array2<f32>> {
    let mut rows = Vec::new();
    let mut ncols = 0;
    
    for item in list.iter() {
        let row: Vec<f32> = item.extract::<&PyList>()?.extract()?;
        if ncols == 0 {
            ncols = row.len();
        } else if row.len() != ncols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All rows must have same length"
            ));
        }
        rows.extend(row);
    }
    
    let nrows = list.len();
    Array2::from_shape_vec((nrows, ncols), rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// convert back to vec
pub fn array2_to_vec<T: Clone>(arr: Array2<T>) -> Vec<Vec<T>> {
    arr.outer_iter()
        .map(|row| row.to_vec())
        .collect()
}
