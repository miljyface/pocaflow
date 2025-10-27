#[cfg(target_os = "macos")]
mod metal_backend;

#[cfg(target_os = "macos")]
pub use metal_backend::MetalContext;

#[cfg(not(target_os = "macos"))]
mod cuda_backend;

#[cfg(not(target_os = "macos"))]
pub use cuda_backend::CudaContext;