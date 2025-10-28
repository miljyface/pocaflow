use cuda_runtime_sys::{
    cudaError_t, cudaStreamCreate, cudaStreamDestroy,
    cudaMalloc, cudaFree, cudaMallocHost, cudaFreeHost,
    cudaMemcpyAsync, cudaMemcpyKind, cudaStreamSynchronize, CUstream_st
};
use ndarray::Array2;
use std::ptr;
use std::ffi::c_void;

// FFI to custom CUDA kernel
extern "C" {
    fn launch_matmul_kernel(
        d_A: *const f32,
        d_B: *const f32,
        d_C: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: *mut CUstream_st,
    );
}

fn cuda_error_to_i32(status: cudaError_t) -> Result<(), i32> {
    if status != cudaError_t::cudaSuccess {
        Err(status as i32)
    } else {
        Ok(())
    }
}

pub struct CudaContext {
    pub stream: *mut CUstream_st,
    pub buffer_a: *mut f32,
    pub buffer_b: *mut f32,
    pub buffer_c: *mut f32,
    pub buffer_size_a: usize,
    pub buffer_size_b: usize,
    pub buffer_size_c: usize,
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
            let mut stream: *mut CUstream_st = ptr::null_mut();
            cuda_error_to_i32(cudaStreamCreate(&mut stream))?;

            let mut buffer_a = ptr::null_mut();
            let mut buffer_b = ptr::null_mut();
            let mut buffer_c = ptr::null_mut();
            let buffer_size_a = max_a_elems * std::mem::size_of::<f32>();
            let buffer_size_b = max_b_elems * std::mem::size_of::<f32>();
            let buffer_size_c = max_c_elems * std::mem::size_of::<f32>();
            cuda_error_to_i32(cudaMalloc(&mut buffer_a as *mut _ as *mut *mut _, buffer_size_a))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_b as *mut _ as *mut *mut _, buffer_size_b))?;
            cuda_error_to_i32(cudaMalloc(&mut buffer_c as *mut _ as *mut *mut _, buffer_size_c))?;

            let mut pinned_a: *mut f32 = ptr::null_mut();
            let mut pinned_b: *mut f32 = ptr::null_mut();
            let mut pinned_c: *mut f32 = ptr::null_mut();
            cuda_error_to_i32(cudaMallocHost(&mut pinned_a as *mut _ as *mut *mut _, buffer_size_a))?;
            cuda_error_to_i32(cudaMallocHost(&mut pinned_b as *mut _ as *mut *mut _, buffer_size_b))?;
            cuda_error_to_i32(cudaMallocHost(&mut pinned_c as *mut _ as *mut *mut _, buffer_size_c))?;

            Ok(CudaContext {
                stream, buffer_a, buffer_b, buffer_c,
                buffer_size_a, buffer_size_b, buffer_size_c,
                pinned_a, pinned_b, pinned_c,
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
            
            if size_a > self.buffer_size_a {
                cudaFree(self.buffer_a as *mut _);
                cuda_error_to_i32(cudaMalloc(&mut self.buffer_a as *mut _ as *mut *mut _, size_a))?;
                self.buffer_size_a = size_a;
                cudaFreeHost(self.pinned_a as *mut _);
                cuda_error_to_i32(cudaMallocHost(&mut self.pinned_a as *mut _ as *mut *mut _, size_a))?;
                self.pinned_size_a = size_a;
            }
            if size_b > self.buffer_size_b {
                cudaFree(self.buffer_b as *mut _);
                cuda_error_to_i32(cudaMalloc(&mut self.buffer_b as *mut _ as *mut *mut _, size_b))?;
                self.buffer_size_b = size_b;
                cudaFreeHost(self.pinned_b as *mut _);
                cuda_error_to_i32(cudaMallocHost(&mut self.pinned_b as *mut _ as *mut *mut _, size_b))?;
                self.pinned_size_b = size_b;
            }
            if size_c > self.buffer_size_c {
                cudaFree(self.buffer_c as *mut _);
                cuda_error_to_i32(cudaMalloc(&mut self.buffer_c as *mut _ as *mut *mut _, size_c))?;
                self.buffer_size_c = size_c;
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
            std::ptr::copy_nonoverlapping(a.as_ptr(), self.pinned_a, a_elems);
            std::ptr::copy_nonoverlapping(b.as_ptr(), self.pinned_b, b_elems);

            cuda_error_to_i32(cudaMemcpyAsync(
                self.buffer_a as *mut c_void, self.pinned_a as *const c_void,
                a_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice, self.stream
            ))?;
            cuda_error_to_i32(cudaMemcpyAsync(
                self.buffer_b as *mut c_void, self.pinned_b as *const c_void,
                b_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice, self.stream
            ))?;

            // Launch custom kernel
            launch_matmul_kernel(
                self.buffer_a, self.buffer_b, self.buffer_c,
                m as i32, n as i32, k as i32, self.stream
            );

            cuda_error_to_i32(cudaMemcpyAsync(
                self.pinned_c as *mut c_void, self.buffer_c as *const c_void,
                c_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, self.stream
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
            cudaStreamDestroy(self.stream);
        }
    }
}
