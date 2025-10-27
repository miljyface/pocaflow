extern crate cublas_sys;
extern crate cuda_sys;

use cublas_sys::*;
use cuda_sys::*;

use std::ptr;
use ndarray::Array2;

pub struct CudaContext {
    pub ctx: CUcontext,
    pub handle: cublasHandle_t,
}

impl CudaContext {
    pub fn new() -> Self {
        unsafe {
            let mut cu_ctx: CUcontext = ptr::null_mut();
            let mut cu_dev: CUdevice = 0;
            cuda_check!(cuInit(0));
            cuda_check!(cuDeviceGet(&mut cu_dev, 0));
            cuda_check!(cuCtxCreate_v2(&mut cu_ctx, 0, cu_dev));
            let mut handle: cublasHandle_t = ptr::null_mut();
            cublasCreate_v2(&mut handle);
            CudaContext { ctx: cu_ctx, handle }
        }
    }

    pub fn matmul_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);

        unsafe {
            let mut d_a: CUdeviceptr = 0;
            let mut d_b: CUdeviceptr = 0;
            let mut d_c: CUdeviceptr = 0;

            let a_bytes = (m * k * std::mem::size_of::<f32>());
            let b_bytes = (k * n * std::mem::size_of::<f32>());
            let c_bytes = (m * n * std::mem::size_of::<f32>());

            cuda_check!(cuMemAlloc_v2(&mut d_a, a_bytes));
            cuda_check!(cuMemAlloc_v2(&mut d_b, b_bytes));
            cuda_check!(cuMemAlloc_v2(&mut d_c, c_bytes));
            cuda_check!(cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const _, a_bytes));
            cuda_check!(cuMemcpyHtoD_v2(d_b, b.as_ptr() as *const _, b_bytes));

            let (lda, ldb, ldc) = (m as i32, k as i32, m as i32);
            let (m, n, k) = (m as i32, n as i32, k as i32);
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            // cublas is column major by default; if row major, swap order
            cublasSgemm(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, // num cols of C (and B)
                m, // num rows of C (and A)
                k,
                &alpha as *const f32,
                d_b as *const f32, n, // B
                d_a as *const f32, k, // A
                &beta as *const f32,
                d_c as *mut f32, n, // C
            );

            let mut c = vec![0.0f32; (m as usize) * (n as usize)];
            cuda_check!(cuMemcpyDtoH_v2(
                c.as_mut_ptr() as *mut _, d_c, c_bytes
            ));

            cuda_check!(cuMemFree_v2(d_a));
            cuda_check!(cuMemFree_v2(d_b));
            cuda_check!(cuMemFree_v2(d_c));

            Array2::from_shape_vec((m as usize, n as usize), c).unwrap()
        }
    }
}
