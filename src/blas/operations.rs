// Thank you ChatGPT

use ndarray::{Array2, ArrayView2};
use super::bindings::*;

/// Safe wrapper for dgemm
#[cfg(target_os = "macos")]
pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    
    let mut result = Array2::<f64>::zeros((m, n));
    
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            result.as_mut_ptr(),
            n as i32,
        );
    }
    
    result
}

/// Safe wrapper for sgemm
#[cfg(target_os = "macos")]
pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    
    let mut result = Array2::<f32>::zeros((m, n));
    
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            result.as_mut_ptr(),
            n as i32,
        );
    }
    
    result
}

// Fallback for non-macOS (Im gonna add shit here for CUDA once I get to it)
#[cfg(not(target_os = "macos"))]
pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    a.dot(&b)
}

#[cfg(not(target_os = "macos"))]
pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    a.dot(&b)
}
