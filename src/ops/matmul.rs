use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, PyAny};
use crate::python::tensor::Tensor;

#[cfg(target_os = "linux")]
use crate::backends::cuda;

#[cfg(target_os = "macos")]
use crate::backends::metal;

// Smart matmul dispatcher - accepts Tensor or converts NumPy to Tensor
#[pyfunction]
pub fn matmul(py: Python, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    // Try to extract as Tensor first (GPU path)
    if let (Ok(a_tensor), Ok(b_tensor)) = (a.extract::<Tensor>(), b.extract::<Tensor>()) {
        // Both are Tensors - use GPU backend
        #[cfg(target_os = "linux")]
        return cuda::matmul::matmul(a_tensor, b_tensor).map(|t| t.into_py(py));
        
        #[cfg(target_os = "macos")]
        return metal::matmul::metal_matmul_f32(a_tensor, b_tensor).map(|t| t.into_py(py));
        
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "GPU not available on this platform"
        ));
    }
    
    // Fall back to NumPy CPU path
    use numpy::{PyArray2, PyReadonlyArray2};
    
    let a_arr: PyReadonlyArray2<f32> = a.extract()?;
    let b_arr: PyReadonlyArray2<f32> = b.extract()?;
    
    use crate::cpu::blas::sgemm;
    let result = sgemm(a_arr.as_array(), b_arr.as_array());
    Ok(PyArray2::from_owned_array(py, result).into_py(py))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    Ok(())
}
