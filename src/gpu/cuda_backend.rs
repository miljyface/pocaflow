use cuda_runtime_sys::*;
use cublas_sys::*;
use ndarray::Array2;
use std::ptr;

#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {{
        let status = $expr;
        if status != cudaSuccess {
            panic!("CUDA error at {}:{}: {:?}", file!(), line!(), status);
        }
    }};
}

pub struct CudaContext {
    pub handle: cublasHandle_t,
    buffer_a: *mut f32,
    buffer_b: *mut f32,
    buffer_c: *mut f32,
    max_elements: usize,
}

impl CudaContext {
    pub fn new() -> Self {
        unsafe {
            let mut handle = ptr::null_mut();
            cublas_check!(cublasCreate_v2(&mut handle));
            
            let max_elements = 16384 * 16384;
            let bytes = (max_elements * std::mem::size_of::<f32>()) as u64;
            
            let mut buffer_a = ptr::null_mut();
            let mut buffer_b = ptr::null_mut();
            let mut buffer_c = ptr::null_mut();
            
            cuda_check!(cudaMalloc(&mut buffer_a as *mut _ as *mut *mut _, bytes));
            cuda_check!(cudaMalloc(&mut buffer_b as *mut _ as *mut *mut _, bytes));
            cuda_check!(cudaMalloc(&mut buffer_c as *mut _ as *mut *mut _, bytes));
            
            CudaContext { 
                handle, 
                buffer_a: buffer_a as *mut f32,
                buffer_b: buffer_b as *mut f32,
                buffer_c: buffer_c as *mut f32,
                max_elements 
            }
        }
    }

    pub fn matmul_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);
        
        unsafe {
            let a_bytes = (m * k * std::mem::size_of::<f32>()) as u64;
            let b_bytes = (k * n * std::mem::size_of::<f32>()) as u64;
            let c_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
            
            cuda_check!(cudaMemcpy(
                self.buffer_a as *mut _, a.as_ptr() as *const _, 
                a_bytes, cudaMemcpyKind_cudaMemcpyHostToDevice
            ));
            cuda_check!(cudaMemcpy(
                self.buffer_b as *mut _, b.as_ptr() as *const _, 
                b_bytes, cudaMemcpyKind_cudaMemcpyHostToDevice
            ));
            
            let alpha = 1.0f32;
            let beta = 0.0f32;
            cublas_check!(cublasSgemm_v2(
                self.handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
                n as i32, m as i32, k as i32, &alpha, 
                self.buffer_b, n as i32,
                self.buffer_a, k as i32, &beta, 
                self.buffer_c, n as i32
            ));
            
            let mut c = vec![0.0f32; m * n];
            cuda_check!(cudaMemcpy(
                c.as_mut_ptr() as *mut _, self.buffer_c as *const _, 
                c_bytes, cudaMemcpyKind_cudaMemcpyDeviceToHost
            ));
            cuda_check!(cudaDeviceSynchronize());
            
            Array2::from_shape_vec((m, n), c).unwrap()
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.buffer_a as *mut _);
            cudaFree(self.buffer_b as *mut _);
            cudaFree(self.buffer_c as *mut _);
            cublasDestroy_v2(self.handle);
        }
    }
}
