use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, Element};
use ndarray::{Array1, Array2};

pub struct PyArray;

impl PyArray {
    #[inline]
    pub fn to_array2_f32(arr: PyReadonlyArray2<f32>) -> Array2<f32> {
        arr.as_array().to_owned()
    }

    #[inline]
    pub fn to_array2_f64(arr: PyReadonlyArray2<f64>) -> Array2<f64> {
        arr.as_array().to_owned()
    }

    #[inline]
    pub fn to_array1_f64(arr: PyReadonlyArray1<f64>) -> Array1<f64> {
        arr.as_array().to_owned()
    }

    #[inline]
    pub fn from_array2_f32<'py>(py: Python<'py>, arr: Array2<f32>) -> &'py PyArray2<f32> {
        PyArray2::from_owned_array(py, arr)
    }

    #[inline]
    pub fn from_array2_f64<'py>(py: Python<'py>, arr: Array2<f64>) -> &'py PyArray2<f64> {
        PyArray2::from_owned_array(py, arr)
    }

    #[inline]
    pub fn from_array1_f64<'py>(py: Python<'py>, arr: Array1<f64>) -> &'py PyArray1<f64> {
        PyArray1::from_owned_array(py, arr)
    }

    #[inline]
    pub fn shape_2d<T: Element>(arr: &PyReadonlyArray2<T>) -> (usize, usize) {
        (arr.shape()[0], arr.shape()[1])
    }

    #[inline]
    pub fn shape_1d<T: Element>(arr: &PyReadonlyArray1<T>) -> usize {
        arr.shape()[0]
    }

    #[inline]
    pub fn is_c_contiguous<T: Element>(arr: &PyReadonlyArray2<T>) -> bool {
        arr.as_array().is_standard_layout()
    }

    #[inline]
    pub unsafe fn as_ptr_f32(arr: &PyReadonlyArray2<f32>) -> *const f32 {
        arr.as_array().as_ptr()
    }

    #[inline]
    pub unsafe fn as_ptr_f64(arr: &PyReadonlyArray2<f64>) -> *const f64 {
        arr.as_array().as_ptr()
    }
}

pub struct ArrayConverter;

impl ArrayConverter {
    pub fn check_2d_shape(a_shape: (usize, usize), b_shape: (usize, usize)) -> Result<(), String> {
        if a_shape.1 != b_shape.0 {
            return Err(format!(
                "Shape mismatch for matmul: ({}, {}) @ ({}, {})",
                a_shape.0, a_shape.1, b_shape.0, b_shape.1
            ));
        }
        Ok(())
    }

    pub fn check_1d_length(a_len: usize, b_len: usize) -> Result<(), String> {
        if a_len != b_len {
            return Err(format!(
                "Length mismatch for dot product: {} != {}",
                a_len, b_len
            ));
        }
        Ok(())
    }
}