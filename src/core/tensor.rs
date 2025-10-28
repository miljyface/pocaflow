use super::{DType, Shape, Stride};
use std::sync::Arc;

/// Device location for tensor data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    #[cfg(target_os = "linux")]
    CUDA(usize), // device id
    #[cfg(target_os = "macos")]
    Metal(usize), // device id
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

/// Storage backend for tensor data
pub enum Storage {
    CPU {
        data: Vec<u8>,
    },
    #[cfg(target_os = "linux")]
    CUDA {
        ptr: *mut std::ffi::c_void,
        size: usize,
        device: usize,
    },
    #[cfg(target_os = "macos")]
    Metal {
        ptr: *mut std::ffi::c_void,
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

    /// Get raw pointer (for GPU operations)
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

/// High-performance tensor with GPU support
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Shape,
    stride: Stride,
    dtype: DType,
    offset: usize,
}

impl Tensor {
    pub fn zeros(shape: Shape, dtype: DType, device: Device) -> Self {
        let numel = shape.numel();
        let size = numel * dtype.size();
        
        let storage = match device {
            Device::CPU => Storage::new_cpu(size),
            #[cfg(target_os = "linux")]
            Device::CUDA(dev_id) => {
                // Allocate on GPU via CUDA
                // This is a placeholder - actual implementation would use cudaMalloc
                Storage::new_cpu(size)
            },
            #[cfg(target_os = "macos")]
            Device::Metal(_dev_id) => {
                // Allocate on GPU via Metal
                Storage::new_cpu(size)
            },
        };

        let stride = Stride::contiguous(&shape);

        Self {
            storage: Arc::new(storage),
            shape,
            stride,
            dtype,
            offset: 0,
        }
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

    /// Get raw data pointer (unsafe, for FFI/GPU ops)
    pub unsafe fn data_ptr(&self) -> *const u8 {
        self.storage.as_ptr().add(self.offset)
    }

    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.stride == Stride::contiguous(&self.shape)
    }

    /// View as slice (CPU only, contiguous only)
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

// Manual Clone for reference-counted storage
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