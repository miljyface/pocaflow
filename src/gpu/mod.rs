#[cfg(target_os = "macos")]
mod backend;

#[cfg(target_os = "macos")]
pub use backend::MetalContext;
