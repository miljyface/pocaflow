pub mod bindings; // Only contains macOS cblas FFI, OK if empty elsewhere
pub mod operations;

// Always re‑export from operations (which defines everything correctly per platform)
pub use operations::{dgemm, sgemm};
