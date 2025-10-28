use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

pub fn sgemm(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2);

    let mut c = Array2::zeros((m, n));
    
    // Collect mutable rows to allow parallel iteration
    let mut rows: Vec<_> = c.axis_iter_mut(Axis(0)).collect();
    
    rows.par_iter_mut().enumerate().for_each(|(i, row)| {
        let a_row = a.row(i);
        for j in 0..n {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                sum += a_row[k_idx] * b[[k_idx, j]];
            }
            row[j] = sum;
        }
    });
    
    c
}

pub fn dgemm(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2);

    let mut c = Array2::zeros((m, n));
    
    let mut rows: Vec<_> = c.axis_iter_mut(Axis(0)).collect();
    
    rows.par_iter_mut().enumerate().for_each(|(i, row)| {
        let a_row = a.row(i);
        for j in 0..n {
            let mut sum = 0.0f64;
            for k_idx in 0..k {
                sum += a_row[k_idx] * b[[k_idx, j]];
            }
            row[j] = sum;
        }
    });
    
    c
}