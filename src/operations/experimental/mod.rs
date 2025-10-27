#[cfg(target_os = "macos")]
pub mod metal_matmul;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod cuda_matmul;
