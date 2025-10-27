use ndarray::{Array2, ArrayView2};

#[cfg(target_os = "macos")]
use super::bindings::*;

// ---- macOS Accelerate versions ----
#[cfg(target_os = "macos")]
#[inline]
pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let n = b.ncols();
    let mut result = Array2::<f64>::zeros((m, n));
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0,
            result.as_mut_ptr(), n as i32,
        );
    }
    result
}

#[cfg(target_os = "macos")]
#[inline]
pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let n = b.ncols();
    let mut result = Array2::<f32>::zeros((m, n));
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a.as_ptr(), k as i32,
            b.as_ptr(), n as i32,
            0.0,
            result.as_mut_ptr(), n as i32,
        );
    }
    result
}

// ---- Cross-platform fallbacks (Linux + Windows) ----
#[cfg(not(target_os = "macos"))]
#[inline]
pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    a.dot(&b)
}

#[cfg(not(target_os = "macos"))]
#[inline]
pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    a.dot(&b)
}
