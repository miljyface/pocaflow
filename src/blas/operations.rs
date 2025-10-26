use ndarray::{Array2, ArrayView2};
use super::bindings::*;

// f64
#[cfg(target_os = "macos")]
pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let (m, k) = (a_owned.nrows(), a_owned.ncols());
    let n = b_owned.ncols();

    let mut result = Array2::<f64>::zeros((m, n));
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a_owned.as_ptr(), k as i32,
            b_owned.as_ptr(), n as i32,
            0.0,
            result.as_mut_ptr(), n as i32,
        );
    }
    result
}

// f32
#[cfg(target_os = "macos")]
pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let (m, k) = (a_owned.nrows(), a_owned.ncols());
    let n = b_owned.ncols();

    let mut result = Array2::<f32>::zeros((m, n));
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
            m as i32, n as i32, k as i32,
            1.0,
            a_owned.as_ptr(), k as i32,
            b_owned.as_ptr(), n as i32,
            0.0,
            result.as_mut_ptr(), n as i32,
        );
    }
    result
}

// fallbacks for non-macos (slow as shit. embarrassing. I need to learn cuda)
#[cfg(not(target_os = "macos"))]
pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    a.to_owned().dot(&b.to_owned())
}

#[cfg(not(target_os = "macos"))]
pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    a.to_owned().dot(&b.to_owned())
}
