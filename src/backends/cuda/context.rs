use cuda_runtime_sys::*;
use cublas_sys::*;
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
    pub streams: Vec<*mut CUstream_st>,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    pub buffer_a: *mut f32,
    pub buffer_b: *mut f32,
    pub buffer_c: *mut f32,
    pub buffer_size_a: usize,
    pub buffer_size_b: usize,
    pub buffer_size_c: usize,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(
        max_a_elems: usize,
        max_b_elems: usize,
        max_c_elems: usize,
        n_streams: usize,
        workspace_size: usize,
    ) -> Result<Self, i32> {
        unsafe {
            let mut handle = ptr::null_mut();
            cublas_error_to_i32(cublasCreate_v2(&mut handle))?;

            let mut streams = Vec::with_capacity(n_streams);
            for _ in 0..n_streams {
                let mut stream: *mut CUstream_st = ptr::null_mut();
                cuda_error_to_i32(cudaStreamCreate(&mut stream))?;
                streams.push(stream);
            }

            let mut workspace: *mut c_void = ptr::null_mut();
            cuda_error_to_i32(cudaMalloc(&mut workspace as *mut *mut c_void, workspace_size))?;

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
                streams,
                workspace,
                workspace_size,
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

    pub fn matmul_f32(&mut self, a: &Array2<f32>, b: &Array2<f32>, stream_idx: usize) -> Result<Array2<f32>, i32> {
        let stream = self.streams[stream_idx % self.streams.len()];
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
            cuda_error_to_i32(cudaMemcpyAsync(
                self.buffer_a as *mut c_void,
                a.as_ptr() as *const c_void,
                a_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;

            cuda_error_to_i32(cudaMemcpyAsync(
                self.buffer_b as *mut c_void,
                b.as_ptr() as *const c_void,
                b_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ))?;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            cublas_error_to_i32(cublasSetStream_v2(self.handle, stream as *mut Struct_CUstream_st))?;

            cublas_error_to_i32(cublasSgemm_v2(
                self.handle,
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32, m as i32, k as i32,
                &alpha,
                self.buffer_b as *const f32, n as i32,
                self.buffer_a as *const f32, k as i32,
                &beta,
                self.buffer_c as *mut f32, n as i32,
            ))?;

            let mut result = vec![0.0f32; c_elems];

            cuda_error_to_i32(cudaMemcpyAsync(
                result.as_mut_ptr() as *mut c_void,
                self.buffer_c as *const c_void,
                c_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            ))?;

            cuda_error_to_i32(cudaStreamSynchronize(stream))?;

            Ok(Array2::from_shape_vec((m, n), result).expect("Shape mismatch"))
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.buffer_a as *mut _);
            cudaFree(self.buffer_b as *mut _);
            cudaFree(self.buffer_c as *mut _);
            cudaFree(self.workspace as *mut _);
            cublasDestroy_v2(self.handle);
            for stream in &self.streams {
                cudaStreamDestroy(*stream);
            }
        }
    }
}