use cuda_runtime_sys::*;
use cublas_sys::*;
use ndarray::{Array2, ShapeBuilder};
use std::ptr;

// Error-checking macros adapted to return Result<(), CudaError>
#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {{
        let status = $expr;
        if status != cudaError::cudaSuccess {
            return Err(status);
        }
    }};
}

#[macro_export]
macro_rules! cublas_check {
    ($expr:expr) => {{
        let status = $expr;
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(status as i32);
        }
    }};
}

pub struct CudaContext {
    pub handle: cublasHandle_t,
    pub stream: cudaStream_t,
}

impl CudaContext {
    pub fn new() -> Result<Self, i32> {
        unsafe {
            let mut handle = ptr::null_mut();
            cublas_check!(cublasCreate_v2(&mut handle))?;

            let mut stream = ptr::null_mut();
            cuda_check!(cudaStreamCreate(&mut stream))?;
            cublas_check!(cublasSetStream(handle, stream))?;
            Ok(CudaContext { handle, stream })
        }
    }

    pub fn matmul_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, i32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);

        unsafe {
            let a_bytes = m * k * std::mem::size_of::<f32>();
            let b_bytes = k * n * std::mem::size_of::<f32>();
            let c_bytes = m * n * std::mem::size_of::<f32>();

            // Allocate device buffers just for needed sizes
            let mut buffer_a = ptr::null_mut();
            let mut buffer_b = ptr::null_mut();
            let mut buffer_c = ptr::null_mut();
            cuda_check!(cudaMalloc(&mut buffer_a as *mut _ as *mut *mut _, a_bytes))?;
            cuda_check!(cudaMalloc(&mut buffer_b as *mut _ as *mut *mut _, b_bytes))?;
            cuda_check!(cudaMalloc(&mut buffer_c as *mut _ as *mut *mut _, c_bytes))?;

            // Allocate pinned (page-locked) host memory for output buffer
            let mut c_host: *mut f32 = ptr::null_mut();
            cuda_check!(cudaMallocHost(&mut c_host as *mut _ as *mut *mut _, c_bytes))?;

            // Asynchronous memory transfers
            cuda_check!(cudaMemcpyAsync(
                buffer_a,
                a.as_ptr() as *const _,
                a_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;
            cuda_check!(cudaMemcpyAsync(
                buffer_b,
                b.as_ptr() as *const _,
                b_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;

            // Matrix multiply: C = A * B; ensure shapes and flags are correct (column-major)
            let alpha = 1.0f32;
            let beta = 0.0f32;
            // WARNING: cuBLAS expects column-major; if ndarray is row-major, swap shape order and use transpose flags.
            cublas_check!(cublasSgemm_v2(
                self.handle,
                cublasOperation_t::CUBLAS_OP_N, // Or _T (transpose) if needed for layout!
                cublasOperation_t::CUBLAS_OP_N,
                n as i32, m as i32, k as i32,
                &alpha,
                buffer_b, n as i32,
                buffer_a, k as i32,
                &beta,
                buffer_c, n as i32,
            ))?;

            // Copy result back asynchronously
            cuda_check!(cudaMemcpyAsync(
                c_host as *mut _,
                buffer_c as *const _,
                c_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            ))?;

            // Wait for all operations to finish
            cuda_check!(cudaStreamSynchronize(self.stream))?;

            let result = std::slice::from_raw_parts(c_host, m * n);
            let output = Array2::from_shape_vec((m, n).strides((n as isize, 1)), result.to_vec()).unwrap();
            cudaFree(buffer_a as *mut _);
            cudaFree(buffer_b as *mut _);
            cudaFree(buffer_c as *mut _);
            cudaFreeHost(c_host as *mut _);

            Ok(output)
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle);
            cudaStreamDestroy(self.stream);
        }
    }
}