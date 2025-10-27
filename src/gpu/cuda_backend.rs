use cuda_sys::*;
use std::ffi::CString;
use std::ptr;
use ndarray::Array2;

// SAFETY
macro_rules! cuda_check {
    ($expr:expr) => {
        let res = $expr;
        assert_eq!(res, 0, "CUDA error: {}", res);
    };
}

// i need to have like 70 matmul names or they'll clash and it makes me want to kms
static PTX: &str = include_str!("matmul_tiled.ptx");

pub struct CudaContext {
    ctx: CUcontext,
    module: CUmodule,
    stream: CUstream,
}

impl CudaContext {
    pub fn new() -> Self {
        unsafe {
            let mut cu_ctx: CUcontext = ptr::null_mut();
            let mut cu_dev: CUdevice = 0;
            let mut cu_stream: CUstream = ptr::null_mut();
            let mut cu_module: CUmodule = ptr::null_mut();
            cuda_check!(cuInit(0));
            cuda_check!(cuDeviceGet(&mut cu_dev, 0));
            cuda_check!(cuCtxCreate_v2(&mut cu_ctx, 0, cu_dev));
            let ptx_cstr = CString::new(PTX).unwrap();
            cuda_check!(cuModuleLoadData(&mut cu_module, ptx_cstr.as_ptr() as *const _));
            cuda_check!(cuStreamCreate(&mut cu_stream, 0));
            CudaContext { ctx: cu_ctx, module: cu_module, stream: cu_stream }
        }
    }

    pub fn matmul_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);

        // dynamic tile sizing for all tensor sizes
        let tile_size = if m.min(n).min(k) >= 64 { 64 }
            else if m.min(n).min(k) >= 32 { 32 }
            else if m.min(n).min(k) >= 16 { 16 }
            else { 8 };

        unsafe {
            let mut d_a: CUdeviceptr = 0;
            let mut d_b: CUdeviceptr = 0;
            let mut d_c: CUdeviceptr = 0;

            let a_bytes = (m * k * std::mem::size_of::<f32>()) as usize;
            let b_bytes = (k * n * std::mem::size_of::<f32>()) as usize;
            let c_bytes = (m * n * std::mem::size_of::<f32>()) as usize;

            cuda_check!(cuMemAlloc_v2(&mut d_a, a_bytes));
            cuda_check!(cuMemAlloc_v2(&mut d_b, b_bytes));
            cuda_check!(cuMemAlloc_v2(&mut d_c, c_bytes));

            cuda_check!(cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const _, a_bytes));
            cuda_check!(cuMemcpyHtoD_v2(d_b, b.as_ptr() as *const _, b_bytes));

            let kernel_name = CString::new("matmul_tiled").unwrap();
            let mut func: CUfunction = ptr::null_mut();
            cuda_check!(cuModuleGetFunction(&mut func, self.module, kernel_name.as_ptr()));

            let grid_x = ((n + tile_size - 1) / tile_size) as u32;
            let grid_y = ((m + tile_size - 1) / tile_size) as u32;
            let block_x = tile_size as u32;
            let block_y = tile_size as u32;
            let shared_mem = std::mem::size_of::<f32>() * tile_size * tile_size * 2;

            let mut params = [
                &d_a as *const _ as *mut _,
                &d_b as *const _ as *mut _,
                &d_c as *const _ as *mut _,
                &(m as i32) as *const _ as *mut _,
                &(n as i32) as *const _ as *mut _,
                &(k as i32) as *const _ as *mut _,
                &(tile_size as i32) as *const _ as *mut _,
            ];

            cuda_check!(cuLaunchKernel(
                func,
                grid_x, grid_y, 1,
                block_x, block_y, 1,
                shared_mem as u32,
                self.stream,
                params.as_mut_ptr(),
                ptr::null_mut()
            ));

            cuda_check!(cuStreamSynchronize(self.stream));

            let mut c = vec![0.0f32; m * n];
            cuda_check!(cuMemcpyDtoH_v2(c.as_mut_ptr() as *mut _, d_c, c_bytes));

            cuda_check!(cuMemFree_v2(d_a));
            cuda_check!(cuMemFree_v2(d_b));
            cuda_check!(cuMemFree_v2(d_c));

            Array2::from_shape_vec((m, n), c).unwrap()
        }
    }
}
