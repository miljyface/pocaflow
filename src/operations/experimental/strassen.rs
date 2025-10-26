use ndarray::{Array2, s};
use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use crate::blas::{dgemm, sgemm, cblas_daxpy, cblas_saxpy};
use rayon::prelude::*;

const STRASSEN_THRESHOLD: usize = 128;
const MAX_PARALLEL_DEPTH: usize = 4;

#[inline]
fn add_matrices_blas_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut result = a.clone();
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_daxpy(n, 1.0, b.as_ptr(), 1, result.as_mut_ptr(), 1);
    }
    result
}

#[inline]
fn sub_matrices_blas_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut result = a.clone();
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_daxpy(n, -1.0, b.as_ptr(), 1, result.as_mut_ptr(), 1);
    }
    result
}

#[inline]
fn add_inplace_f64(a: &mut Array2<f64>, b: &Array2<f64>) {
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_daxpy(n, 1.0, b.as_ptr(), 1, a.as_mut_ptr(), 1);
    }
}

#[inline]
fn sub_inplace_f64(a: &mut Array2<f64>, b: &Array2<f64>) {
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_daxpy(n, -1.0, b.as_ptr(), 1, a.as_mut_ptr(), 1);
    }
}

// f32 versions
#[inline]
fn add_matrices_blas_f32(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut result = a.clone();
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_saxpy(n, 1.0, b.as_ptr(), 1, result.as_mut_ptr(), 1);
    }
    result
}

#[inline]
fn sub_matrices_blas_f32(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut result = a.clone();
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_saxpy(n, -1.0, b.as_ptr(), 1, result.as_mut_ptr(), 1);
    }
    result
}

#[inline]
fn add_inplace_f32(a: &mut Array2<f32>, b: &Array2<f32>) {
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_saxpy(n, 1.0, b.as_ptr(), 1, a.as_mut_ptr(), 1);
    }
}

#[inline]
fn sub_inplace_f32(a: &mut Array2<f32>, b: &Array2<f32>) {
    let n = (a.nrows() * a.ncols()) as i32;
    unsafe {
        cblas_saxpy(n, -1.0, b.as_ptr(), 1, a.as_mut_ptr(), 1);
    }
}

// fuck this shit
fn strassen_f64_internal(a: &Array2<f64>, b: &Array2<f64>, depth: usize) -> Array2<f64> {
    let n = a.nrows();

    if n <= STRASSEN_THRESHOLD {
        return dgemm(a.view(), b.view());
    }

    let new_size = n.next_power_of_two();
    let needs_padding = new_size != n;
    
    let (a_work, b_work) = if needs_padding {
        let mut a_pad = Array2::<f64>::zeros((new_size, new_size));
        let mut b_pad = Array2::<f64>::zeros((new_size, new_size));
        a_pad.slice_mut(s![..n, ..n]).assign(a);
        b_pad.slice_mut(s![..n, ..n]).assign(b);
        (a_pad, b_pad)
    } else {
        (a.clone(), b.clone())
    };

    let half = new_size / 2;

    let a11 = a_work.slice(s![..half, ..half]);
    let a12 = a_work.slice(s![..half, half..]);
    let a21 = a_work.slice(s![half.., ..half]);
    let a22 = a_work.slice(s![half.., half..]);

    let b11 = b_work.slice(s![..half, ..half]);
    let b12 = b_work.slice(s![..half, half..]);
    let b21 = b_work.slice(s![half.., ..half]);
    let b22 = b_work.slice(s![half.., half..]);

    let s1 = sub_matrices_blas_f64(&b12.to_owned(), &b22.to_owned());
    let s2 = add_matrices_blas_f64(&a11.to_owned(), &a12.to_owned());
    let s3 = add_matrices_blas_f64(&a21.to_owned(), &a22.to_owned());
    let s4 = sub_matrices_blas_f64(&b21.to_owned(), &b11.to_owned());
    let s5 = add_matrices_blas_f64(&a11.to_owned(), &a22.to_owned());
    let s6 = add_matrices_blas_f64(&b11.to_owned(), &b22.to_owned());
    let s7 = sub_matrices_blas_f64(&a12.to_owned(), &a22.to_owned());
    let s8 = add_matrices_blas_f64(&b21.to_owned(), &b22.to_owned());
    let s9 = sub_matrices_blas_f64(&a11.to_owned(), &a21.to_owned());
    let s10 = add_matrices_blas_f64(&b11.to_owned(), &b12.to_owned());

    let (m1, m2, m3, m4, m5, m6, m7) = if depth < MAX_PARALLEL_DEPTH {
        let products = vec![
            (s5, s6),
            (s3, b11.to_owned()),
            (a11.to_owned(), s1),
            (a22.to_owned(), s4),
            (s2, b22.to_owned()),
            (s9, s10),
            (s7, s8),
        ];
        
        let results: Vec<Array2<f64>> = products
            .into_par_iter()
            .map(|(x, y)| strassen_f64_internal(&x, &y, depth + 1))
            .collect();

        (
            results[0].clone(),
            results[1].clone(),
            results[2].clone(),
            results[3].clone(),
            results[4].clone(),
            results[5].clone(),
            results[6].clone(),
        )
    } else {
        // sequential
        let m1 = strassen_f64_internal(&s5, &s6, depth + 1);
        let m2 = strassen_f64_internal(&s3, &b11.to_owned(), depth + 1);
        let m3 = strassen_f64_internal(&a11.to_owned(), &s1, depth + 1);
        let m4 = strassen_f64_internal(&a22.to_owned(), &s4, depth + 1);
        let m5 = strassen_f64_internal(&s2, &b22.to_owned(), depth + 1);
        let m6 = strassen_f64_internal(&s9, &s10, depth + 1);
        let m7 = strassen_f64_internal(&s7, &s8, depth + 1);

        (m1, m2, m3, m4, m5, m6, m7)
    };

    // combine results with in-place operations
    let mut c11 = add_matrices_blas_f64(&m1, &m4);
    sub_inplace_f64(&mut c11, &m5);
    add_inplace_f64(&mut c11, &m7);

    let c12 = add_matrices_blas_f64(&m3, &m5);
    let c21 = add_matrices_blas_f64(&m2, &m4);

    let mut c22 = add_matrices_blas_f64(&m1, &m3);
    sub_inplace_f64(&mut c22, &m2);
    add_inplace_f64(&mut c22, &m6);

    // assemble result
    let mut result = Array2::<f64>::zeros((new_size, new_size));
    result.slice_mut(s![..half, ..half]).assign(&c11);
    result.slice_mut(s![..half, half..]).assign(&c12);
    result.slice_mut(s![half.., ..half]).assign(&c21);
    result.slice_mut(s![half.., half..]).assign(&c22);

    if needs_padding {
        result.slice(s![..n, ..n]).to_owned()
    } else {
        result
    }
}

fn strassen_f32_internal(a: &Array2<f32>, b: &Array2<f32>, depth: usize) -> Array2<f32> {
    let n = a.nrows();

    if n <= STRASSEN_THRESHOLD {
        return sgemm(a.view(), b.view());
    }

    let new_size = n.next_power_of_two();
    let needs_padding = new_size != n;
    
    let (a_work, b_work) = if needs_padding {
        let mut a_pad = Array2::<f32>::zeros((new_size, new_size));
        let mut b_pad = Array2::<f32>::zeros((new_size, new_size));
        a_pad.slice_mut(s![..n, ..n]).assign(a);
        b_pad.slice_mut(s![..n, ..n]).assign(b);
        (a_pad, b_pad)
    } else {
        (a.clone(), b.clone())
    };

    let half = new_size / 2;

    let a11 = a_work.slice(s![..half, ..half]);
    let a12 = a_work.slice(s![..half, half..]);
    let a21 = a_work.slice(s![half.., ..half]);
    let a22 = a_work.slice(s![half.., half..]);

    let b11 = b_work.slice(s![..half, ..half]);
    let b12 = b_work.slice(s![..half, half..]);
    let b21 = b_work.slice(s![half.., ..half]);
    let b22 = b_work.slice(s![half.., half..]);

    let s1 = sub_matrices_blas_f32(&b12.to_owned(), &b22.to_owned());
    let s2 = add_matrices_blas_f32(&a11.to_owned(), &a12.to_owned());
    let s3 = add_matrices_blas_f32(&a21.to_owned(), &a22.to_owned());
    let s4 = sub_matrices_blas_f32(&b21.to_owned(), &b11.to_owned());
    let s5 = add_matrices_blas_f32(&a11.to_owned(), &a22.to_owned());
    let s6 = add_matrices_blas_f32(&b11.to_owned(), &b22.to_owned());
    let s7 = sub_matrices_blas_f32(&a12.to_owned(), &a22.to_owned());
    let s8 = add_matrices_blas_f32(&b21.to_owned(), &b22.to_owned());
    let s9 = sub_matrices_blas_f32(&a11.to_owned(), &a21.to_owned());
    let s10 = add_matrices_blas_f32(&b11.to_owned(), &b12.to_owned());

    let (m1, m2, m3, m4, m5, m6, m7) = if depth < MAX_PARALLEL_DEPTH {
        let products = vec![
            (s5, s6), (s3, b11.to_owned()), (a11.to_owned(), s1),
            (a22.to_owned(), s4), (s2, b22.to_owned()), (s9, s10), (s7, s8),
        ];
        
        let results: Vec<Array2<f32>> = products
            .into_par_iter()
            .map(|(x, y)| strassen_f32_internal(&x, &y, depth + 1))
            .collect();

        (results[0].clone(), results[1].clone(), results[2].clone(),
         results[3].clone(), results[4].clone(), results[5].clone(), results[6].clone())
    } else {
        let m1 = strassen_f32_internal(&s5, &s6, depth + 1);
        let m2 = strassen_f32_internal(&s3, &b11.to_owned(), depth + 1);
        let m3 = strassen_f32_internal(&a11.to_owned(), &s1, depth + 1);
        let m4 = strassen_f32_internal(&a22.to_owned(), &s4, depth + 1);
        let m5 = strassen_f32_internal(&s2, &b22.to_owned(), depth + 1);
        let m6 = strassen_f32_internal(&s9, &s10, depth + 1);
        let m7 = strassen_f32_internal(&s7, &s8, depth + 1);
        (m1, m2, m3, m4, m5, m6, m7)
    };

    let mut c11 = add_matrices_blas_f32(&m1, &m4);
    sub_inplace_f32(&mut c11, &m5);
    add_inplace_f32(&mut c11, &m7);

    let c12 = add_matrices_blas_f32(&m3, &m5);
    let c21 = add_matrices_blas_f32(&m2, &m4);

    let mut c22 = add_matrices_blas_f32(&m1, &m3);
    sub_inplace_f32(&mut c22, &m2);
    add_inplace_f32(&mut c22, &m6);

    let mut result = Array2::<f32>::zeros((new_size, new_size));
    result.slice_mut(s![..half, ..half]).assign(&c11);
    result.slice_mut(s![..half, half..]).assign(&c12);
    result.slice_mut(s![half.., ..half]).assign(&c21);
    result.slice_mut(s![half.., half..]).assign(&c22);

    if needs_padding {
        result.slice(s![..n, ..n]).to_owned()
    } else {
        result
    }
}

#[pyfunction]
pub fn strassen_matmul_f64<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let a_arr = a.as_array().to_owned();
    let b_arr = b.as_array().to_owned();
    let result = strassen_f64_internal(&a_arr, &b_arr, 0);
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
pub fn strassen_matmul_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let a_arr = a.as_array().to_owned();
    let b_arr = b.as_array().to_owned();
    let result = strassen_f32_internal(&a_arr, &b_arr, 0);
    Ok(PyArray2::from_owned_array(py, result))
}

#[pyfunction]
pub fn strassen_matmul(py: Python, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    if let Ok(a_f64) = a.extract::<PyReadonlyArray2<f64>>() {
        if let Ok(b_f64) = b.extract::<PyReadonlyArray2<f64>>() {
            return strassen_matmul_f64(py, a_f64, b_f64).map(|arr| arr.into_py(py));
        }
    }
    if let Ok(a_f32) = a.extract::<PyReadonlyArray2<f32>>() {
        if let Ok(b_f32) = b.extract::<PyReadonlyArray2<f32>>() {
            return strassen_matmul_f32(py, a_f32, b_f32).map(|arr| arr.into_py(py));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "strassen_matmul requires float32 or float64 arrays",
    ))
}
