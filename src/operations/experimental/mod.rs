#[cfg(target_os = "macos")]
pub mod metal_matmul;

#[cfg(target_os = "linux")]
pub mod cuda_matmul;
