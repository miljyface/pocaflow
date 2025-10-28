use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray2, PyReadonlyArray2};
use crate::gpu::CudaContext;
use std::cell::RefCell;

thread_local! {
    static CUDA_CTX: RefCell<Option<CudaContext>> = RefCell::new(None);
}

#[pyfunction]
pub fn cuda_matmul_f32<'py>(
    py: Python<'py>,
    a: &PyAny,
    b: &PyAny,
) -> PyResult<PyObject> {
    // Check if inputs are already CUDA tensors
    let a_device = a.getattr("device")?.str()?.to_string();
    let b_device = b.getattr("device")?.str()?.to_string();
    
    if a_device.contains("cuda") && b_device.contains("cuda") {
        // FAST PATH: Already on GPU, use PyTorch's matmul
        let torch = py.import("torch")?;
        let result = torch.call_method1("matmul", (a, b))?;
        return Ok(result.into());
    }
    
    // SLOW PATH: CPU tensors, need to copy
    let a_array = a.extract::<PyReadonlyArray2<f32>>()?;
    let b_array = b.extract::<PyReadonlyArray2<f32>>()?;
    let a_owned = a_array.as_array().to_owned();
    let b_owned = b_array.as_array().to_owned();

    let m = a_owned.shape()[0];
    let k = a_owned.shape()[1];
    let n = b_owned.shape()[1];
    let max_a_elems = m * k;
    let max_b_elems = k * n;
    let max_c_elems = m * n;

    let result = CUDA_CTX.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        if ctx_ref.is_none() {
            *ctx_ref = Some(CudaContext::new(max_a_elems, max_b_elems, max_c_elems).map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to init CUDA context: {}", e)
                )
            )?);
        }
        ctx_ref.as_mut().unwrap().ensure_capacity(max_a_elems, max_b_elems, max_c_elems).map_err(|e|
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("CUDA ensure_capacity failed: {}", e)
            )
        )?;
        ctx_ref.as_mut().unwrap().matmul_f32(&a_owned, &b_owned)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("CUDA matmul failed: {}", e)
            ))
    })?;
    
    Ok(PyArray2::from_owned_array(py, result).into())
}
