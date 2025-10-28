use cuda_runtime_sys::*;
use std::ptr;
use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;

// cuBLAS-LT FFI bindings
#[link(name = "cublasLt")]
extern "C" {
    fn cublasLtCreate(handle: *mut *mut c_void) -> i32;
    fn cublasLtDestroy(handle: *mut c_void) -> i32;
    fn cublasLtMatmul(
        handle: *mut c_void,
        computeDesc: *mut c_void,
        alpha: *const c_void,
        A: *const c_void, Adesc: *mut c_void,
        B: *const c_void, Bdesc: *mut c_void,
        beta: *const c_void,
        C: *mut c_void, Cdesc: *mut c_void,
        D: *mut c_void, Ddesc: *mut c_void,
        algo: *const c_void,
        workspace: *mut c_void,
        workspaceSize: usize,
        stream: cudaStream_t,
    ) -> i32;
    fn cublasLtMatmulDescCreate(desc: *mut *mut c_void, computeType: i32, scaleType: i32) -> i32;
    fn cublasLtMatrixLayoutCreate(layout: *mut *mut c_void, dtype: i32, rows: u64, cols: u64, ld: u64) -> i32;
}

const CUBLAS_COMPUTE_32F_FAST_TF32: i32 = 4119;
const CUDA_R_32F: i32 = 0;

type CudaStream = *mut CUstream_st;

pub struct CudaGraphCache {
    graphs: HashMap<(usize, usize, usize), (*mut c_void, *mut c_void)>,
}

impl CudaGraphCache {
    pub fn new() -> Self {
        CudaGraphCache {
            graphs: HashMap::new(),
        }
    }

    pub fn get_or_create(
        &mut self,
        m: usize, n: usize, k: usize,
        create_fn: impl FnOnce() -> (*mut c_void, *mut c_void)
    ) -> (*mut c_void, *mut c_void) {
        *self.graphs.entry((m, n, k)).or_insert_with(create_fn)
    }
}

pub struct CudaContext {
    pub handle: *mut c_void,
    pub streams: Vec<CudaStream>,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    pub graph_cache: Mutex<CudaGraphCache>,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(n_streams: usize, workspace_size: usize) -> Result<Self, String> {
        unsafe {
            let mut handle = ptr::null_mut();
            if cublasLtCreate(&mut handle) != 0 {
                return Err("Failed to create cuBLASLt handle".into());
            }

            let mut streams = Vec::with_capacity(n_streams);
            for _ in 0..n_streams {
                let mut stream = ptr::null_mut();
                if cudaStreamCreate(&mut stream) != cudaError_t::cudaSuccess {
                    return Err("Failed to create CUDA stream".into());
                }
                streams.push(stream);
            }

            let mut workspace = ptr::null_mut();
            if cudaMalloc(&mut workspace, workspace_size) != cudaError_t::cudaSuccess {
                return Err("Failed to allocate workspace".into());
            }

            Ok(CudaContext {
                handle,
                streams,
                workspace,
                workspace_size,
                graph_cache: Mutex::new(CudaGraphCache::new()),
            })
        }
    }

    pub fn matmul_f32_gpu(
        &mut self,
        d_a: *const f32, d_b: *const f32, d_c: *mut f32,
        m: usize, n: usize, k: usize,
        stream_idx: usize
    ) -> Result<(), String> {
        let stream = self.streams[stream_idx % self.streams.len()];
        
        unsafe {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            let mut computeDesc = ptr::null_mut();
            if cublasLtMatmulDescCreate(
                &mut computeDesc,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUDA_R_32F
            ) != 0 {
                return Err("Failed to create matmul descriptor".into());
            }

            let mut Adesc = ptr::null_mut();
            let mut Bdesc = ptr::null_mut();
            let mut Cdesc = ptr::null_mut();

            cublasLtMatrixLayoutCreate(&mut Adesc, CUDA_R_32F, k as u64, m as u64, k as u64);
            cublasLtMatrixLayoutCreate(&mut Bdesc, CUDA_R_32F, n as u64, k as u64, n as u64);
            cublasLtMatrixLayoutCreate(&mut Cdesc, CUDA_R_32F, n as u64, m as u64, n as u64);

            let status = cublasLtMatmul(
                self.handle,
                computeDesc,
                &alpha as *const f32 as *const c_void,
                d_a as *const c_void, Adesc,
                d_b as *const c_void, Bdesc,
                &beta as *const f32 as *const c_void,
                d_c as *mut c_void, Cdesc,
                d_c as *mut c_void, Cdesc,
                ptr::null(),
                self.workspace,
                self.workspace_size,
                stream,
            );

            cudaStreamSynchronize(stream);

            if status != 0 {
                Err(format!("cuBLASLt matmul failed: {}", status))
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
            for &stream in &self.streams {
                cudaStreamDestroy(stream);
            }
            cublasLtDestroy(self.handle);
        }
    }
}
