use super::{DType, Shape, Stride};
use std::sync::Arc;
use std::ptr;

#[cfg(target_os = "linux")]
use cuda_runtime_sys::{cudaMalloc, cudaFree, cudaMemset, cudaMemcpy, 
                       cudaMemcpyKind, cudaError_t};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    #[cfg(target_os = "linux")]
    CUDA(usize),
    #[cfg(target_os = "macos")]
    Metal(usize),
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Device::CPU)
    }

    pub fn id(&self) -> usize {
        match self {
            Device::CPU => 0,
            #[cfg(target_os = "linux")]
            Device::CUDA(id) => *id,
            #[cfg(target_os = "macos")]
            Device::Metal(id) => *id,
        }
    }
}

pub enum Storage {
    CPU {
        data: Vec<u8>,
    },
    #[cfg(target_os = "linux")]
    CUDA {
        ptr: *mut c_void,
        size: usize,
        device: usize,
    },
    #[cfg(target_os = "macos")]
    Metal {
        ptr: *mut c_void,
        size: usize,
        device: usize,
    },
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    pub fn new_cpu(size: usize) -> Self {
        Storage::CPU {
            data: vec![0u8; size],
        }
    }

    #[cfg(target_os = "linux")]
    pub fn new_cuda(size: usize, device: usize) -> Result<Self, String> {
        unsafe {
            let ret = cuda_runtime_sys::cudaSetDevice(device as i32);
            if ret != cudaError_t::cudaSuccess {
                return Err(format!("cudaSetDevice failed: {:?}", ret));
            }

            let mut ptr: *mut c_void = ptr::null_mut();
            let ret = cudaMalloc(&mut ptr, size);
            if ret != cudaError_t::cudaSuccess || ptr.is_null() {
                return Err(format!("cudaMalloc failed: {:?}", ret));
            }

            cudaMemset(ptr, 0, size);
            Ok(Storage::CUDA { ptr, size, device })
        }
    }

    #[cfg(target_os = "macos")]
    pub fn new_metal(size: usize, device: usize) -> Result<Self, String> {
        use metal::{Device as MetalDevice, MTLResourceOptions};
        
        let mtl_device = MetalDevice::system_default()
            .ok_or("Failed to get Metal device")?;
        
        let buffer = mtl_device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let ptr = buffer.contents() as *mut c_void;
        
        unsafe {
            std::ptr::write_bytes(ptr as *mut u8, 0, size);
        }
        
        std::mem::forget(buffer);
        Ok(Storage::Metal { ptr, size, device })
    }

    pub fn size(&self) -> usize {
        match self {
            Storage::CPU { data } => data.len(),
            #[cfg(target_os = "linux")]
            Storage::CUDA { size, .. } => *size,
            #[cfg(target_os = "macos")]
            Storage::Metal { size, .. } => *size,
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::CPU { .. } => Device::CPU,
            #[cfg(target_os = "linux")]
            Storage::CUDA { device, .. } => Device::CUDA(*device),
            #[cfg(target_os = "macos")]
            Storage::Metal { device, .. } => Device::Metal(*device),
        }
    }

    pub unsafe fn as_ptr(&self) -> *const u8 {
        match self {
            Storage::CPU { data } => data.as_ptr(),
            #[cfg(target_os = "linux")]
            Storage::CUDA { ptr, .. } => *ptr as *const u8,
            #[cfg(target_os = "macos")]
            Storage::Metal { ptr, .. } => *ptr as *const u8,
        }
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Storage::CPU { data } => data.as_mut_ptr(),
            #[cfg(target_os = "linux")]
            Storage::CUDA { ptr, .. } => *ptr as *mut u8,
            #[cfg(target_os = "macos")]
            Storage::Metal { ptr, .. } => *ptr as *mut u8,
        }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        match self {
            Storage::CPU { .. } => {}
            #[cfg(target_os = "linux")]
            Storage::CUDA { ptr, .. } => unsafe {
                if !ptr.is_null() {
                    cudaFree(*ptr);
                }
            },
            #[cfg(target_os = "macos")]
            Storage::Metal { .. } => {}
        }
    }
}

pub struct Tensor {
    storage: Arc<Storage>,
    shape: Shape,
    stride: Stride,
    dtype: DType,
    offset: usize,
}

impl Tensor {
    pub fn zeros(shape: Shape, dtype: DType, device: Device) -> Result<Self, String> {
        let numel = shape.numel();
        let size = numel * dtype.size();
        
        let storage = match device {
            Device::CPU => Storage::new_cpu(size),
            #[cfg(target_os = "linux")]
            Device::CUDA(dev_id) => Storage::new_cuda(size, dev_id)?,
            #[cfg(target_os = "macos")]
            Device::Metal(dev_id) => Storage::new_metal(size, dev_id)?,
        };

        let stride = Stride::contiguous(&shape);

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            stride,
            dtype,
            offset: 0,
        })
    }

    pub fn from_slice_f32(data: &[f32], shape: Shape) -> Self {
        assert_eq!(data.len(), shape.numel());
        let size = data.len() * std::mem::size_of::<f32>();
        let mut bytes = vec![0u8; size];
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                size,
            );
        }

        let stride = Stride::contiguous(&shape);

        Self {
            storage: Arc::new(Storage::CPU { data: bytes }),
            shape,
            stride,
            dtype: DType::F32,
            offset: 0,
        }
    }

    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    #[inline]
    pub fn stride(&self) -> &Stride {
        &self.stride
    }

    pub unsafe fn data_ptr(&self) -> *const u8 {
        self.storage.as_ptr().add(self.offset)
    }

    pub fn is_contiguous(&self) -> bool {
        self.stride == Stride::contiguous(&self.shape)
    }

    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 || self.device() != Device::CPU || !self.is_contiguous() {
            return None;
        }

        match &*self.storage {
            Storage::CPU { data } => {
                let ptr = unsafe { data.as_ptr().add(self.offset) as *const f32 };
                Some(unsafe { std::slice::from_raw_parts(ptr, self.numel()) })
            },
            _ => None,
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            dtype: self.dtype,
            offset: self.offset,
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device())
            .finish()
    }
}

// GPU-specific tensor for zero-copy operations
#[cfg(target_os = "linux")]
pub struct GpuTensor {
    pub ptr: *mut f32,
    pub shape: (usize, usize),
    pub device: usize,
}

#[cfg(target_os = "linux")]
impl GpuTensor {
    pub fn zeros(m: usize, n: usize) -> Result<Self, String> {
        unsafe {
            let mut ptr: *mut f32 = ptr::null_mut();
            let size = m * n * std::mem::size_of::<f32>();
            if cudaMalloc(&mut ptr as *mut _ as *mut *mut c_void, size) != cudaError_t::cudaSuccess {
                return Err("cudaMalloc failed".into());
            }
            cudaMemset(ptr as *mut c_void, 0, size);
            Ok(GpuTensor { ptr, shape: (m, n), device: 0 })
        }
    }

    pub fn from_host(data: &[f32], m: usize, n: usize) -> Result<Self, String> {
        let mut tensor = Self::zeros(m, n)?;
        unsafe {
            let size = m * n * std::mem::size_of::<f32>();
            cudaMemcpy(
                tensor.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                cudaMemcpyKind::cudaMemcpyHostToDevice
            );
        }
        Ok(tensor)
    }

    pub fn to_host(&self) -> Vec<f32> {
        let (m, n) = self.shape;
        let mut result = vec![0.0f32; m * n];
        unsafe {
            cudaMemcpy(
                result.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                m * n * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost
            );
        }
        result
    }
}

#[cfg(target_os = "linux")]
impl Drop for GpuTensor {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr as *mut c_void); }
    }
}
