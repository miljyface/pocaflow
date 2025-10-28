#[cfg(target_os = "linux")]
pub mod cuda;

#[cfg(target_os = "macos")]
pub mod metal;