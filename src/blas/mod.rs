#[cfg(target_os = "macos")]
pub mod bindings;

pub mod operations;

#[cfg(target_os = "macos")]
pub use operations::{dgemm, sgemm}; // only available with Accelerate
