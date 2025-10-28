use cuda_runtime_sys::{
    cudaError_t, cudaStreamCreate, cudaStreamDestroy,
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
    pub buffer_a: *mut f32,
    pub buffer_b: *mut f32,
    pub buffer_c: *mut f32,
    pub buffer_size_a: usize,
    pub buffer_size_b: usize,
    pub buffer_size_c: usize,
}

impl CudaContext {
    pub fn new(max_a_elems: usize, max_b_elems: usize, max_c_elems: usize) -> Result<Self, i32> {
        unsafe {
            let mut handle = ptr::null_mut();
            cublas_error_to_i32(cublasCreate_v2(&mut handle))?;

            let mut stream: *mut CUstream_st = ptr::null_mut();
            cuda_error_to_i32(cudaStreamCreate(&mut stream))?;

            cublas_error_to_i32(cublasSetStream_v2(
                handle,
                stream as *mut Struct_CUstream_st,
            ))?;

            let mut buffer_a = ptr::null_mut();
            let mut buffer_b = ptr::null_mut();
            let mut buffer_c = ptr::null_mut();
            let buffer_size_a = max_a_elems * std::mem::size_of::<f32>();
            let buffer_size_b = max_b_elems * std::mem::size_of::<f32>();
            let buffer_size_c = max_c_elems * std::mem::size_of::<f32>();
            cuda_error_to_i32(cudaMalloc(&mut buffer_a as *mut _ as *mut *mut _, buffer_size_a))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_b as *mut _ as *mut *mut _, buffer_size_b))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_c as *mut _ as *mut *mut _, buffer_size_c))?;

            Ok(CudaContext {
                handle,
                stream,
                buffer_a,
                buffer_b,
                buffer_c,
                buffer_size_a,
                buffer_size_b,
                buffer_size_c,
            })
        }
    }

    pub fn ensure_capacity(&mut self, need_a: usize, need_b: usize, need_c: usize) -> Result<(), i32> {
        unsafe {
            let size_a = need_a * std::mem::size_of::<f32>();
            let size_b = need_b * std::mem::size_of::<f32>();
            let size_c = need_c * std::mem::size_of::<f32>();
            if size_a > self.buffer_size_a {
                cudaFree(self.buffer_a as *mut _);
                cuda_error_to_i32(cudaMalloc(&mut self.buffer_a as *mut _ as *mut *mut _, size_a))?;
                self.buffer_size_a = size_a;
            }
            if size_b > self.buffer_size_b {
                cudaFree(self.buffer_b as *mut _);
                cuda_error_to_i32(cudaMalloc(&mut self.buffer_b as *mut _ as *mut *mut _, size_b))?;
                self.buffer_size_b = size_b;
            }
            if size_c > self.buffer_size_c {
                cudaFree(self.buffer_c as *mut _);
                cuda_error_to_i32(cudaMalloc(&mut self.buffer_c as *mut _ as *mut *mut _, size_c))?;
                self.buffer_size_c = size_c;
            }
            Ok(())
        }
    }

    pub fn matmul_f32(&mut self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, i32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);
        let a_elems = m * k;
        let b_elems = k * n;
        let c_elems = m * n;
        let a_bytes = a_elems * std::mem::size_of::<f32>();
        let b_bytes = b_elems * std::mem::size_of::<f32>();
        let c_bytes = c_elems * std::mem::size_of::<f32>();

        // Ensure persistent GPU buffer is large enough
        self.ensure_capacity(a_elems, b_elems, c_elems)?;

        // Allocate pinned host memory for inputs and output
        let mut pinned_a: *mut f32 = ptr::null_mut();
        let mut pinned_b: *mut f32 = ptr::null_mut();
        let mut pinned_c: *mut f32 = ptr::null_mut();
        unsafe {
            cuda_error_to_i32(cudaMallocHost(&mut pinned_a as *mut _ as *mut *mut _, a_bytes))?;
            cuda_error_to_i32(cudaMallocHost(&mut pinned_b as *mut _ as *mut *mut _, b_bytes))?;
            cuda_error_to_i32(cudaMallocHost(&mut pinned_c as *mut _ as *mut *mut _, c_bytes))?;
            std::ptr::copy_nonoverlapping(a.as_ptr(), pinned_a, a_elems);
            std::ptr::copy_nonoverlapping(b.as_ptr(), pinned_b, b_elems);

            cuda_error_to_i32(cudaMemcpyAsync(self.buffer_a, pinned_a, a_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice, self.stream))?;
            cuda_error_to_i32(cudaMemcpyAsync(self.buffer_b, pinned_b, b_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice, self.stream))?;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            cublas_error_to_i32(
                cublasSgemm_v2(
                    self.handle,
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_T,
                    m as i32, n as i32, k as i32,
                    &alpha,
                    self.buffer_a as *const f32, k as i32,
                    self.buffer_b as *const f32, n as i32,
                    &beta,
                    self.buffer_c as *mut f32, m as i32,
                )
            )?;

            cuda_error_to_i32(cudaMemcpyAsync(pinned_c, self.buffer_c, c_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, self.stream))?;
            cuda_error_to_i32(cudaStreamSynchronize(self.stream))?;

            let result = std::slice::from_raw_parts(pinned_c, c_elems);
            let output = Array2::from_shape_vec((m, n), result.to_vec()).expect("CUDA result shape mismatch");

            cudaFreeHost(pinned_a as *mut _);
            cudaFreeHost(pinned_b as *mut _);
            cudaFreeHost(pinned_c as *mut _);

            Ok(output)
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
            cudaStreamDestroy(self.stream);
        }
    }
}
