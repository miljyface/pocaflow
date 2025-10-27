mod bindings;
mod operations;

// src/blas/mod.rs
#[cfg(target_os = "macos")]
pub use operations::{dgemm, sgemm};
#[cfg(not(target_os = "macos"))]
// On linux, import from openblas or manually declare stubs/erroring items.

// src/operations/matmul.rs
#[cfg(target_os = "macos")]
use super::experimental::metal_matmul::cuda_matmul_f32 as gpu_matmul_f32;
#[cfg(not(target_os = "macos"))]
// Provide a dummy or alternative implementation for Linux:

// src/operations/experimental/cuda_matmul.rs
#[cfg(target_os = "macos")]
use crate::gpu::CudaContext;
// On Linux, either provide another context or conditionally skip this module.
