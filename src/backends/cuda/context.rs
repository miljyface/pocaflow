use cuda_runtime_sys::*;
use std::ptr;
use std::ffi::c_void;
use std::collections::VecDeque;
use std::sync::Mutex;

#[repr(C)]
pub struct cublasLtHandle_st { _unused: [u8; 0] }
pub type cublasLtHandle_t = *mut cublasLtHandle_st;
#[repr(C)]
pub struct cublasLtMatmulDesc_st { _unused: [u8; 0] }
pub type cublasLtMatmulDesc_t = *mut cublasLtMatmulDesc_st;
#[repr(C)]
pub struct cublasLtMatrixLayout_st { _unused: [u8; 0] }
pub type cublasLtMatrixLayout_t = *mut cublasLtMatrixLayout_st;

const CUBLASLT_MATMUL_DESC_TRANSA: u32 = 0;
const CUBLASLT_MATMUL_DESC_TRANSB: u32 = 1;
const CUBLAS_OP_N: u8 = 0;

extern "C" {
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        sizeInBytes: usize,
    ) -> i32;
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtDestroy(handle: cublasLtHandle_t) -> i32;
    fn cublasLtMatmul(
        lightHandle: cublasLtHandle_t, computeDesc: cublasLtMatmulDesc_t, alpha: *const c_void,
        A: *const c_void, Adesc: cublasLtMatrixLayout_t, B: *const c_void, Bdesc: cublasLtMatrixLayout_t,
        beta: *const c_void, C: *const c_void, Cdesc: cublasLtMatrixLayout_t,
        D: *mut c_void, Ddesc: cublasLtMatrixLayout_t, algo: *const c_void,
        workspace: *mut c_void, workspaceSizeInBytes: usize, stream: cudaStream_t,
    ) -> i32;
    fn cublasLtMatmulDescCreate(desc: *mut cublasLtMatmulDesc_t, computeType: i32, scaleType: i32) -> i32;
    fn cublasLtMatrixLayoutCreate(layout: *mut cublasLtMatrixLayout_t, type_: i32, rows: u64, cols: u64, ld: u64) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
}

const CUBLAS_COMPUTE_32F: i32 = 68;
const CUDA_R_32F: i32 = 0;
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
    pub handle: cublasLtHandle_t,
    pub streams: Vec<CudaStream>,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    pub mem_pool: Mutex<MemoryPool>,
}
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(n_streams: usize, workspace_size: usize) -> Result<Self, String> {
        unsafe {
            let mut handle = ptr::null_mut();
            if cublasLtCreate(&mut handle) != 0 {
                return Err("cublasLtCreate failed".into());
            }
            let mut streams = Vec::new();
            for _ in 0..n_streams {
                let mut stream = ptr::null_mut();
                if cudaStreamCreate(&mut stream) != cudaError_t::cudaSuccess {
                    return Err("cudaStreamCreate failed".into());
                }
                streams.push(stream);
            }
            let mut workspace = ptr::null_mut();
            if cudaMalloc(&mut workspace, workspace_size) != cudaError_t::cudaSuccess {
                return Err("workspace malloc failed".into());
            }
            Ok(CudaContext {
                handle,
                streams,
                workspace,
                workspace_size,
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
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let mut operationDesc = ptr::null_mut();
            if cublasLtMatmulDescCreate(&mut operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != 0 {
                return Err("matmulDescCreate failed".into());
            }
            // Explicitly set op(A) and op(B) to "no transpose"
            let op_none: u8 = 0;
            if cublasLtMatmulDescSetAttribute(
                operationDesc,
                CUBLASLT_MATMUL_DESC_TRANSA,
                &op_none as *const _ as *const c_void,
                1,
            ) != 0 {
                return Err("MatmulDescSetAttribute TRANSA failed".into());
            }
            if cublasLtMatmulDescSetAttribute(
                operationDesc,
                CUBLASLT_MATMUL_DESC_TRANSB,
                &op_none as *const _ as *const c_void,
                1,
            ) != 0 {
                return Err("MatmulDescSetAttribute TRANSB failed".into());
            }
            let mut Adesc = ptr::null_mut();
            let mut Bdesc = ptr::null_mut();
            let mut Cdesc = ptr::null_mut();
            cublasLtMatrixLayoutCreate(&mut Adesc, CUDA_R_32F, m as u64, k as u64, m as u64);
            cublasLtMatrixLayoutCreate(&mut Bdesc, CUDA_R_32F, k as u64, n as u64, k as u64);
            cublasLtMatrixLayoutCreate(&mut Cdesc, CUDA_R_32F, m as u64, n as u64, m as u64);
            println!("A: {:p}, shape=({},{}), ld={}", d_a, m, k, m);
            println!("B: {:p}, shape=({},{}), ld={}", d_b, k, n, k);
            println!("C: {:p}, shape=({},{}), ld={}", d_c, m, n, m);
            let status = cublasLtMatmul(
                self.handle, operationDesc,
                &alpha as *const f32 as *const c_void,
                d_a as *const c_void, Adesc,
                d_b as *const c_void, Bdesc,
                &beta as *const f32 as *const c_void,
                d_c as *const c_void, Cdesc,
                d_c as *mut c_void, Cdesc,
                ptr::null(), self.workspace, self.workspace_size, stream,
            );
            cudaStreamSynchronize(stream);
            cublasLtMatmulDescDestroy(operationDesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            if status != 0 {
                Err(format!("cublasLtMatmul failed: {}", status))
            } else {
                Ok(())
            }
        }
    }
}
impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.workspace);
            for &s in &self.streams { cudaStreamDestroy(s); }
            cublasLtDestroy(self.handle);
        }
    }
}
