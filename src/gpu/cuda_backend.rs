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
use std::ffi::c_void;

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
    // Device buffers (persistent)
    pub buffer_a: *mut f32,
    pub buffer_b: *mut f32,
    pub buffer_c: *mut f32,
    pub buffer_size_a: usize,
    pub buffer_size_b: usize,
    pub buffer_size_c: usize,
    // Pinned host buffers (persistent)
    pub pinned_a: *mut f32,
    pub pinned_b: *mut f32,
    pub pinned_c: *mut f32,
    pub pinned_size_a: usize,
    pub pinned_size_b: usize,
    pub pinned_size_c: usize,
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

            // Allocate device buffers
            let mut buffer_a = ptr::null_mut();
            let mut buffer_b = ptr::null_mut();
            let mut buffer_c = ptr::null_mut();
            let buffer_size_a = max_a_elems * std::mem::size_of::<f32>();
            let buffer_size_b = max_b_elems * std::mem::size_of::<f32>();
            let buffer_size_c = max_c_elems * std::mem::size_of::<f32>();
            cuda_error_to_i32(cudaMalloc(&mut buffer_a as *mut _ as *mut *mut _, buffer_size_a))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_b as *mut _ as *mut *mut _, buffer_size_b))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_c as *mut _ as *mut *mut _, buffer_size_c))?;

            // Allocate pinned host buffers (persistent!)
            let mut pinned_a: *mut f32 = ptr::null_mut();
            let mut pinned_b: *mut f32 = ptr::null_mut();
            let mut pinned_c: *mut f32 = ptr::null_mut();
            cuda_error_to_i32(cudaMallocHost(&mut pinned_a as *mut _ as *mut *mut _, buffer_size_a))?;
            cuda_error_to_i32(cudaMallocHost(&mut pinned_b as *mut _ as *mut *mut _, buffer_size_b))?;
            cuda_error_to_i32(cudaMallocHost(&mut pinned_c as *mut _ as *mut *mut _, buffer_size_c))?;

            Ok(CudaContext {
                handle,
                stream,
                buffer_a,
                buffer_b,
                buffer_c,
                buffer_size_a,
                buffer_size_b,
                buffer_size_c,
                pinned_a,
                pinned_b,
                pinned_c,
                pinned_size_a: buffer_size_a,
                pinned_size_b: buffer_size_b,
                pinned_size_c: buffer_size_c,
            })
        }
    }

    pub fn ensure_capacity(&mut self, need_a: usize, need_b: usize, need_c: usize) -> Result<(), i32> {
        unsafe {
            let size_a = need_a * std::mem::size_of::<f32>();
            let size_b = need_b * std::mem::size_of::<f32>();
            let size_c = need_c * std::mem::size_of::<f32>();
            
            // Resize device buffers if needed
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

            // Resize pinned host buffers if needed
            if size_a > self.pinned_size_a {
                cudaFreeHost(self.pinned_a as *mut _);
                cuda_error_to_i32(cudaMallocHost(&mut self.pinned_a as *mut _ as *mut *mut _, size_a))?;
                self.pinned_size_a = size_a;
            }
            if size_b > self.pinned_size_b {
                cudaFreeHost(self.pinned_b as *mut _);
                cuda_error_to_i32(cudaMallocHost(&mut self.pinned_b as *mut _ as *mut *mut _, size_b))?;
                self.pinned_size_b = size_b;
            }
            if size_c > self.pinned_size_c {
                cudaFreeHost(self.pinned_c as *mut _);
                cuda_error_to_i32(cudaMallocHost(&mut self.pinned_c as *mut _ as *mut *mut _, size_c))?;
                self.pinned_size_c = size_c;
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

        self.ensure_capacity(a_elems, b_elems, c_elems)?;

        unsafe {
            // Copy input data to persistent pinned buffers
            std::ptr::copy_nonoverlapping(a.as_ptr(), self.pinned_a, a_elems);
            std::ptr::copy_nonoverlapping(b.as_ptr(), self.pinned_b, b_elems);

            // Async transfer to device
            cuda_error_to_i32(cudaMemcpyAsync(
                self.buffer_a as *mut c_void,
                self.pinned_a as *const c_void,
                a_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream
            ))?;
            cuda_error_to_i32(cudaMemcpyAsync(
                self.buffer_b as *mut c_void,
                self.pinned_b as *const c_void,
                b_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream
            ))?;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            // CRITICAL FIX: Use proper row-major to column-major mapping
            // For row-major A (m x k) and B (k x n) computing C = A * B (m x n)
            // We call cuBLAS as: C' = B' * A' where ' means column-major interpretation
            // This avoids extra transposes and matches PyTorch's optimized path
            cublas_error_to_i32(cublasSgemm_v2(
                self.handle,
                cublasOperation_t::CUBLAS_OP_N,  // B is not transposed
                cublasOperation_t::CUBLAS_OP_N,  // A is not transposed
                n as i32,  // number of rows of B' (columns of B row-major)
                m as i32,  // number of columns of A' (rows of A row-major)
                k as i32,  // shared dimension
                &alpha,
                self.buffer_b as *const f32, n as i32,  // B', leading dimension n
                self.buffer_a as *const f32, k as i32,  // A', leading dimension k
                &beta,
                self.buffer_c as *mut f32, n as i32,    // C', leading dimension n
            ))?;

            // Async transfer back to host
            cuda_error_to_i32(cudaMemcpyAsync(
                self.pinned_c as *mut c_void,
                self.buffer_c as *const c_void,
                c_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream
            ))?;
            cuda_error_to_i32(cudaStreamSynchronize(self.stream))?;

            let result = std::slice::from_raw_parts(self.pinned_c, c_elems);
            let output = Array2::from_shape_vec((m, n), result.to_vec())
                .expect("CUDA result shape mismatch");

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
            cudaFreeHost(self.pinned_a as *mut _);
            cudaFreeHost(self.pinned_b as *mut _);
            cudaFreeHost(self.pinned_c as *mut _);
            cublasDestroy_v2(self.handle);
            cudaStreamDestroy(self.stream);
        }
    }
}
