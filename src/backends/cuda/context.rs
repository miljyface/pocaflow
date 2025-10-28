use cuda_runtime_sys::*;
use std::ptr;
use std::ffi::c_void;
use std::collections::HashMap;
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

#[link(name = "cublasLt")]
extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtDestroy(handle: cublasLtHandle_t) -> i32;
    fn cublasLtMatmul(
        lightHandle: cublasLtHandle_t, computeDesc: cublasLtMatmulDesc_t,
        alpha: *const c_void, A: *const c_void, Adesc: cublasLtMatrixLayout_t,
        B: *const c_void, Bdesc: cublasLtMatrixLayout_t, beta: *const c_void,
        C: *const c_void, Cdesc: cublasLtMatrixLayout_t,
        D: *mut c_void, Ddesc: cublasLtMatrixLayout_t,
        algo: *const c_void, workspace: *mut c_void, workspaceSizeInBytes: usize,
        stream: cudaStream_t,
    ) -> i32;
    fn cublasLtMatmulDescCreate(matmulDesc: *mut cublasLtMatmulDesc_t, computeType: i32, scaleType: i32) -> i32;
    fn cublasLtMatmulDescSetAttribute(matmulDesc: cublasLtMatmulDesc_t, attr: i32, buf: *const c_void, sizeInBytes: usize) -> i32;
    fn cublasLtMatrixLayoutCreate(matLayout: *mut cublasLtMatrixLayout_t, type_: i32, rows: u64, cols: u64, ld: u64) -> i32;
}

const CUBLAS_COMPUTE_32F: i32 = 68;
const CUDA_R_32F: i32 = 0;
const CUBLASLT_MATMUL_DESC_TRANSA: i32 = 0;
const CUBLASLT_MATMUL_DESC_TRANSB: i32 = 1;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_OP_N: i32 = 0;

type CudaStream = *mut CUstream_st;

pub struct BufferCache {
    buffers: HashMap<(usize, usize), *mut f32>,
}

impl BufferCache {
    pub fn new() -> Self {
        BufferCache { buffers: HashMap::new() }
    }
    
    pub fn get_or_alloc(&mut self, m: usize, n: usize) -> *mut f32 {
        *self.buffers.entry((m, n)).or_insert_with(|| unsafe {
            let mut ptr: *mut f32 = ptr::null_mut();
            cudaMalloc(&mut ptr as *mut _ as *mut *mut c_void, m * n * 4);
            ptr
        })
    }
}

pub struct CudaContext {
    pub handle: cublasLtHandle_t,
    pub streams: Vec<CudaStream>,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    pub buffer_cache: Mutex<BufferCache>,
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

            Ok(CudaContext { handle, streams, workspace, workspace_size, buffer_cache: Mutex::new(BufferCache::new()) })
        }
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

            // Set transpose for A (OP_T)
            let transa = CUBLAS_OP_T;
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                &transa as *const i32 as *const c_void, 4);
            
            // Set transpose for B (OP_N)
            let transb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                &transb as *const i32 as *const c_void, 4);

            let mut Adesc = ptr::null_mut();
            let mut Bdesc = ptr::null_mut();
            let mut Cdesc = ptr::null_mut();

            // A transposed: k x m, stored as m x k with ld=m
            cublasLtMatrixLayoutCreate(&mut Adesc, CUDA_R_32F, k as u64, m as u64, m as u64);
            // B normal: k x n, stored with ld=k
            cublasLtMatrixLayoutCreate(&mut Bdesc, CUDA_R_32F, k as u64, n as u64, k as u64);
            // C: m x n, stored with ld=m
            cublasLtMatrixLayoutCreate(&mut Cdesc, CUDA_R_32F, m as u64, n as u64, m as u64);

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
