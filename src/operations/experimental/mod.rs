#[cfg(target_os = "macos")]
pub mod metal_matmul;

#[cfg(not(target_os = "macos"))]
pub mod cuda_matmul;
