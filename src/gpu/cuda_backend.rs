use cuda_runtime_sys::{
    cudaError_t, cudaStream_t, cudaStreamCreate, cudaStreamDestroy,
    cudaMalloc, cudaFree, cudaMallocHost, cudaFreeHost,
    cudaMemcpyAsync, cudaMemcpyKind, cudaStreamSynchronize, CUstream_st
};
use cublas_sys::{
    cublasHandle_t, cublasCreate_v2, cublasDestroy_v2,
    cublasSgemm_v2, cublasSetStream_v2,
    cublasOperation_t, cublasStatus_t, Struct_CUstream_st
};
use ndarray::Array2;
use std::ptr;

// Error helpers
fn cuda_error_to_i32(status: cudaError_t) -> Result<(), i32> {
    if status != cudaError_t::cudaSuccess {
        Err(status as i32)
    } else {
        Ok(())
    }
}

fn cublas_error_to_i32(status: cublasStatus_t) -> Result<(), i32> {
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Err(status as i32)
    } else {
        Ok(())
    }
}

pub struct CudaContext {
    pub handle: cublasHandle_t,
    pub stream: *mut CUstream_st,
}

impl CudaContext {
    pub fn new() -> Result<Self, i32> {
        unsafe {
            let mut handle = ptr::null_mut();
            cublas_error_to_i32(cublasCreate_v2(&mut handle))?;

            let mut stream: *mut CUstream_st = ptr::null_mut();
            cuda_error_to_i32(cudaStreamCreate(&mut stream))?;

            // Must cast CUstream_st to Struct_CUstream_st for cuBLAS
            cublas_error_to_i32(cublasSetStream_v2(
                handle,
                stream as *mut Struct_CUstream_st,
            ))?;

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

            let mut buffer_a = ptr::null_mut();
            let mut buffer_b = ptr::null_mut();
            let mut buffer_c = ptr::null_mut();
            cuda_error_to_i32(cudaMalloc(&mut buffer_a as *mut _ as *mut *mut _, a_bytes))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_b as *mut _ as *mut *mut _, b_bytes))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_c as *mut _ as *mut *mut _, c_bytes))?;

            let mut c_host: *mut f32 = ptr::null_mut();
            cuda_error_to_i32(cudaMallocHost(&mut c_host as *mut _ as *mut *mut _, c_bytes))?;

            cuda_error_to_i32(cudaMemcpyAsync(
                buffer_a,
                a.as_ptr() as *const _,
                a_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;
            cuda_error_to_i32(cudaMemcpyAsync(
                buffer_b,
                b.as_ptr() as *const _,
                b_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            // cuBLAS expects column-major, but ndarray is row-major.
            // Use transpose flags to reinterpret row-major as column-major.
            cublas_error_to_i32(
                cublasSgemm_v2(
                    self.handle,
                    cublasOperation_t::CUBLAS_OP_T, // transposes a
                    cublasOperation_t::CUBLAS_OP_T, // transposes b
                    m as i32, n as i32, k as i32,
                    &alpha,
                    buffer_a as *const f32, k as i32,
                    buffer_b as *const f32, n as i32,
                    &beta,
                    buffer_c as *mut f32, m as i32,
                )
            )?;

            cuda_error_to_i32(cudaMemcpyAsync(
                c_host as *mut _,
                buffer_c as *const _,
                c_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            ))?;
            cuda_error_to_i32(cudaStreamSynchronize(self.stream))?;

            let result = std::slice::from_raw_parts(c_host, m * n);
            let output = Array2::from_shape_vec((m, n), result.to_vec()).expect("CUDA result shape mismatch");

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
