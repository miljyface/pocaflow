#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod metal_backend;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use metal_backend::MetalContext;

#[cfg(not(target_os = "macos"))]
mod cuda_backend;

#[cfg(not(target_os = "macos"))]
pub use cuda_backend::CudaContext;