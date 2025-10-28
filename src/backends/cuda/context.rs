use cuda_runtime_sys::*;
use std::ptr;
use std::ffi::c_void;
use std::collections::VecDeque;
use std::sync::Mutex;

#[repr(C)]
pub struct cublasContext { _unused: [u8; 0] }
pub type cublasHandle_t = *mut cublasContext;

#[repr(i32)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
}

#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> i32;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> i32;
    fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudaStream_t) -> i32;
    fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: *const f32,
        C: *mut f32,
        ldc: i32,
    ) -> i32;
}

type CudaStream = *mut CUstream_st;

pub struct MemoryPool {
    pool: VecDeque<(*mut f32, usize, usize)>,
    max_size: usize,
}
impl MemoryPool {
    fn new(max_size: usize) -> Self {
        MemoryPool { pool: VecDeque::new(), max_size }
    }
    fn get(&mut self, m: usize, n: usize) -> *mut f32 {
        for i in 0..self.pool.len() {
            let (ptr, pm, pn) = self.pool[i];
            if pm == m && pn == n {
                self.pool.remove(i);
                return ptr;
            }
        }
        unsafe {
            let mut ptr: *mut f32 = ptr::null_mut();
            cudaMalloc(&mut ptr as *mut _ as *mut *mut c_void, m * n * 4);
            ptr
        }
    }
    fn release(&mut self, ptr: *mut f32, m: usize, n: usize) {
        if self.pool.len() < self.max_size {
            self.pool.push_back((ptr, m, n));
        } else {
            unsafe { cudaFree(ptr as *mut c_void); }
        }
    }
}
impl Drop for MemoryPool {
    fn drop(&mut self) {
        unsafe {
            for (ptr, _, _) in &self.pool {
                cudaFree(*ptr as *mut c_void);
            }
        }
    }
}

pub struct CudaContext {
    pub handle: cublasHandle_t,
    pub streams: Vec<CudaStream>,
    pub mem_pool: Mutex<MemoryPool>,
}
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(n_streams: usize, _workspace_size: usize) -> Result<Self, String> {
        unsafe {
            let mut handle = ptr::null_mut();
            if cublasCreate_v2(&mut handle) != 0 {
                return Err("cublasCreate failed".into());
            }
            let mut streams = Vec::new();
            for _ in 0..n_streams {
                let mut stream = ptr::null_mut();
                if cudaStreamCreate(&mut stream) != cudaError_t::cudaSuccess {
                    return Err("cudaStreamCreate failed".into());
                }
                streams.push(stream);
            }
            Ok(CudaContext {
                handle,
                streams,
                mem_pool: Mutex::new(MemoryPool::new(10)),
            })
        }
    }
    pub fn alloc(&self, m: usize, n: usize) -> *mut f32 {
        self.mem_pool.lock().unwrap().get(m, n)
    }
    pub fn free(&self, ptr: *mut f32, m: usize, n: usize) {
        self.mem_pool.lock().unwrap().release(ptr, m, n);
    }
    pub fn matmul_f32_gpu(&mut self, d_a: *const f32, d_b: *const f32, d_c: *mut f32,
                         m: usize, n: usize, k: usize, stream_idx: usize) -> Result<(), String> {
        let stream = self.streams[stream_idx % self.streams.len()];
        unsafe {
            cublasSetStream_v2(self.handle, stream);
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            // cuBLAS Fortran-order gemm: C = alpha * A * B + beta * C
            let status = cublasSgemm_v2(
                self.handle,
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                d_a,
                m as i32, // lda = m for Fortran column-major
                d_b,
                k as i32, // ldb = k
                &beta,
                d_c,
                m as i32, // ldc = m
            );
            cudaStreamSynchronize(stream);
            if status != 0 {
                Err(format!("cublasSgemm failed: {}", status))
            } else {
                Ok(())
            }
        }
    }
}
impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            for &s in &self.streams { cudaStreamDestroy(s); }
            cublasDestroy_v2(self.handle);
        }
    }
}
