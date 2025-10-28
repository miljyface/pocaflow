use cuda_runtime_sys::{
    cudaError_t, cudaStreamCreate, cudaStreamDestroy,
    cudaMalloc, cudaFree, cudaMallocHost, cudaFreeHost,
    cudaMemcpyAsync, cudaMemcpyKind, cudaStreamSynchronize, CUstream_st,
    cudaGraphCreate, cudaGraphLaunch, cudaGraphExec_t, cudaGraphDestroy,
    cudaGraphExecDestroy
};
use cublas_sys::{
    cublasHandle_t, cublasCreate_v2, cublasDestroy_v2,
    cublasSetMathMode, cublasMath_t,
    cublasLtHandle_t, cublasLtCreate, cublasLtDestroy,
    cublasLtMatmul, cublasLtMatmulDesc_t, cublasLtMatrixLayoutDesc_t,
    cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
    cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
    cublasLtMatmulDescSetAttribute, cublasLtMatrixLayoutSetAttribute,
    cublasOperation_t, cublasStatus_t, cublasLtPointerMode_t,
    CUBLASLT_MATRIX_LAYOUT_TYPE, CUBLASLT_MATRIX_LAYOUT_ROWS,
    cublasLtOrder_t, CUBLASLT_ORDER_ROW, 
    cublasSetStream_v2, Struct_CUstream_st,
    cublasComputeType_t, CUBLAS_COMPUTE_32F_FAST_TF32
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
    pub lt_handle: cublasLtHandle_t,
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
    // CUDA Graph (captured once, replayed many times)
    pub graph: cudaGraphExec_t,
    pub graph_m: i32,
    pub graph_n: i32,
    pub graph_k: i32,
}

impl CudaContext {
    pub fn new(max_a_elems: usize, max_b_elems: usize, max_c_elems: usize) -> Result<Self, i32> {
        unsafe {
            let mut handle = ptr::null_mut();
            cublas_error_to_i32(cublasCreate_v2(&mut handle))?;

            let mut lt_handle = ptr::null_mut();
            cublas_error_to_i32(cublasLtCreate(&mut lt_handle))?;

            let mut stream: *mut CUstream_st = ptr::null_mut();
            cuda_error_to_i32(cudaStreamCreate(&mut stream))?;
            cublas_error_to_i32(cublasSetStream_v2(
                handle,
                stream as *mut Struct_CUstream_st,
            ))?;

            // Enable TF32 for faster FP32 on Ampere and newer (opt-in fast mode)
            let _ = cublasSetMathMode(handle, cublasMath_t::CUBLAS_DEFAULT_MATH);

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

            // Initialize a dummy graph (will be replaced on first matmul)
            let mut dummy_graph = ptr::null_mut();
            let _ = cudaGraphCreate(&mut dummy_graph, 0);
            let graph_exec = std::mem::zeroed(); // Placeholder

            Ok(CudaContext {
                handle,
                lt_handle,
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
                graph: graph_exec,
                graph_m: 0,
                graph_n: 0,
                graph_k: 0,
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

            // Use cuBLASLt for automatic algorithm tuning (better for variable sizes)
            let mut mat_a = ptr::null_mut();
            let mut mat_b = ptr::null_mut();
            let mut mat_c = ptr::null_mut();
            let mut matmul_desc = ptr::null_mut();

            cublas_error_to_i32(cublasLtMatrixLayoutCreate(&mut mat_a, 32u32, m as i64, k as i64, k as i64))?;
            cublas_error_to_i32(cublasLtMatrixLayoutCreate(&mut mat_b, 32u32, k as i64, n as i64, n as i64))?;
            cublas_error_to_i32(cublasLtMatrixLayoutCreate(&mut mat_c, 32u32, m as i64, n as i64, n as i64))?;
            cublas_error_to_i32(cublasLtMatmulDescCreate(&mut matmul_desc, cublasComputeType_t::CUBLAS_COMPUTE_32F))?;

            // Set row-major layout for all matrices (ndarray is row-major)
            cublas_error_to_i32(cublasLtMatrixLayoutSetAttribute(
                mat_a,
                CUBLASLT_MATRIX_LAYOUT_TYPE,
                &(cublasLtOrder_t::CUBLASLT_ORDER_ROW as u32) as *const _ as *const c_void,
                std::mem::size_of::<u32>()
            ))?;
            cublas_error_to_i32(cublasLtMatrixLayoutSetAttribute(
                mat_b,
                CUBLASLT_MATRIX_LAYOUT_TYPE,
                &(cublasLtOrder_t::CUBLASLT_ORDER_ROW as u32) as *const _ as *const c_void,
                std::mem::size_of::<u32>()
            ))?;
            cublas_error_to_i32(cublasLtMatrixLayoutSetAttribute(
                mat_c,
                CUBLASLT_MATRIX_LAYOUT_TYPE,
                &(cublasLtOrder_t::CUBLASLT_ORDER_ROW as u32) as *const _ as *const c_void,
                std::mem::size_of::<u32>()
            ))?;

            // Call cuBLASLt
            cublas_error_to_i32(cublasLtMatmul(
                self.lt_handle,
                matmul_desc,
                &alpha as *const f32 as *const c_void,
                self.buffer_a as *const c_void,
                mat_a,
                self.buffer_b as *const c_void,
                mat_b,
                &beta as *const f32 as *const c_void,
                self.buffer_c as *const c_void,
                mat_c,
                self.buffer_c as *mut c_void,
                mat_c,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                self.stream
            ))?;

            cublasLtMatmulDescDestroy(matmul_desc);
            cublasLtMatrixLayoutDestroy(mat_a);
            cublasLtMatrixLayoutDestroy(mat_b);
            cublasLtMatrixLayoutDestroy(mat_c);

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
            cublasLtDestroy(self.lt_handle);
            cudaStreamDestroy(self.stream);
        }
    }
}
