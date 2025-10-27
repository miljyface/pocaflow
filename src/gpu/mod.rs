#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod metal_backend;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use metal_backend::MetalContext;

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
mod cuda_backend;

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
pub use cuda_backend::CudaContext;