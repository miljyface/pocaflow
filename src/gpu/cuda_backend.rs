#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {
        {
            let status = $expr;
            if status != cuda_sys::CUresult::CUDA_SUCCESS {
                panic!("CUDA error at {}:{}: {:?}", file!(), line!(), status);
            }
        }
    };
}

#[macro_export]
macro_rules! cublas_check {
    ($expr:expr) => {
        {
            let status = $expr;
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                panic!("cuBLAS error at {}:{}: {:?}", file!(), line!(), status);
            }
        }
    };
}

// --- Imports (bring all needed types/functions into scope) ---
use cublas_sys::{
    cublasHandle_t, cublasCreate_v2, cublasDestroy_v2, cublasSetStream_v2, 
    cublasSgemm, cublasStatus_t, CUBLAS_OP_N
};
use cuda_sys::{
    CUcontext, CUdevice, CUstream, CUdeviceptr, cuInit, cuDeviceGet, cuCtxCreate_v2, cuStreamCreate,
    cuMemAlloc_v2, cuMemcpyHtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuStreamSynchronize, cuStreamDestroy_v2,
    cuMemFree_v2, cuCtxDestroy_v2
};
use std::ptr;
use ndarray::Array2;

// ---- CUDA Context struct ----
pub struct CudaContext {
    pub ctx: CUcontext,
    pub handle: cublasHandle_t,
    pub stream: CUstream,
    buffer_a: CUdeviceptr,
    buffer_b: CUdeviceptr,
    buffer_c: CUdeviceptr,
    max_elements: usize,
}

impl CudaContext {
    pub fn new() -> Self {
        unsafe {
            let mut cu_ctx: CUcontext = ptr::null_mut();
            let mut cu_dev: CUdevice = 0;

            crate::cuda_check!(cuInit(0));
            crate::cuda_check!(cuDeviceGet(&mut cu_dev, 0));
            crate::cuda_check!(cuCtxCreate_v2(&mut cu_ctx, 0, cu_dev));

            let mut handle: cublasHandle_t = ptr::null_mut();
            crate::cublas_check!(cublasCreate_v2(&mut handle));

            let mut stream = ptr::null_mut();
            crate::cuda_check!(cuStreamCreate(&mut stream, 0));
            crate::cublas_check!(cublasSetStream_v2(handle, stream));

            let max_elements = 16384 * 16384;
            let bytes = max_elements * std::mem::size_of::<f32>();

            let mut buffer_a = 0;
            let mut buffer_b = 0;
            let mut buffer_c = 0;
            crate::cuda_check!(cuMemAlloc_v2(&mut buffer_a, bytes));
            crate::cuda_check!(cuMemAlloc_v2(&mut buffer_b, bytes));
            crate::cuda_check!(cuMemAlloc_v2(&mut buffer_c, bytes));

            CudaContext {
                ctx: cu_ctx,
                handle,
                stream,
                buffer_a,
                buffer_b,
                buffer_c,
                max_elements,
            }
        }
    }

    pub fn matmul_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);
        assert!(m * k <= self.max_elements);
        assert!(k * n <= self.max_elements);
        assert!(m * n <= self.max_elements);
        
        unsafe {
            let a_bytes = m * k * std::mem::size_of::<f32>();
            let b_bytes = k * n * std::mem::size_of::<f32>();
            let c_bytes = m * n * std::mem::size_of::<f32>();
            
            // Copy to GPU
            cuda_check!(cuMemcpyHtoDAsync_v2(
                self.buffer_a, 
                a.as_ptr() as *const _, 
                a_bytes,
                self.stream
            ));
            cuda_check!(cuMemcpyHtoDAsync_v2(
                self.buffer_b, 
                b.as_ptr() as *const _, 
                b_bytes,
                self.stream
            ));
            
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            
            // Row-major: compute C = A*B via C^T = B^T * A^T
            cublas_check!(cublasSgemm(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,  // rows of B^T
                m as i32,  // cols of A^T
                k as i32,  // shared dimension
                &alpha as *const f32,
                self.buffer_b as *const f32, n as i32,  // B
                self.buffer_a as *const f32, k as i32,  // A
                &beta as *const f32,
                self.buffer_c as *mut f32, n as i32,    // C
            ));
            
            // Copy result back
            let mut c = vec![0.0f32; m * n];
            cuda_check!(cuMemcpyDtoHAsync_v2(
                c.as_mut_ptr() as *mut _,
                self.buffer_c,
                c_bytes,
                self.stream
            ));
            
            // Wait for completion
            cuda_check!(cuStreamSynchronize(self.stream));
            
            Array2::from_shape_vec((m, n), c).unwrap()
        }
    }
}

// ---- Proper resource cleanup ----
impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cuStreamDestroy_v2(self.stream);
            cublasDestroy_v2(self.handle);
            cuMemFree_v2(self.buffer_a);
            cuMemFree_v2(self.buffer_b);
            cuMemFree_v2(self.buffer_c);
            cuCtxDestroy_v2(self.ctx);
        }
    }
}
