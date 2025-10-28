use pyo3::prelude::*;
use crate::core::{Tensor as CoreTensor, Shape, DType};
use numpy::PyArray2;

#[pyclass]
pub struct Tensor {
    inner: CoreTensor,
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        let dtype = match dtype {
            "f32" | "float32" => DType::F32,
            "f64" | "float64" => DType::F64,
            "i32" | "int32" => DType::I32,
            "i64" | "int64" => DType::I64,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid dtype")),
        };

        let shape = Shape::new(shape);
        let device = crate::core::tensor::Device::CPU;
        
        Ok(Self {
            inner: CoreTensor::zeros(shape, dtype, device)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?,
        })
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<Self> {
        Self::new(shape, dtype)
    }

    #[staticmethod]
    fn from_array(_py: Python, arr: &PyAny) -> PyResult<Self> {
        if let Ok(arr_f32) = arr.extract::<numpy::PyReadonlyArray2<f32>>() {
            let shape = Shape::new(vec![arr_f32.shape()[0], arr_f32.shape()[1]]);
            let data: Vec<f32> = arr_f32.as_array().iter().copied().collect();
            
            Ok(Self {
                inner: CoreTensor::from_slice_f32(&data, shape),
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Expected f32 array"))
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    fn dtype(&self) -> &str {
        self.inner.dtype().name()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={:?})",
            self.inner.shape().dims(),
            self.inner.dtype(),
            self.inner.device()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.shape().dims()[0]
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        if let Some(data) = self.inner.as_slice_f32() {
            let shape = self.inner.shape().dims();
            if shape.len() == 2 {
                let arr = ndarray::Array2::from_shape_vec(
                    (shape[0], shape[1]),
                    data.to_vec(),
                ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                
                Ok(PyArray2::from_owned_array(py, arr).into())
            } else {
                Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Only 2D tensors supported for now"
                ))
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot convert GPU tensor to numpy"
            ))
        }
    }
}

pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    Ok(())
}

