pub mod matmul;
pub mod batch;
pub mod vec_ops;
mod conversion;  // Only this internal module

// Re-export conversion functions for internal use
pub(crate) use conversion::{pylist_to_array2_f64, pylist_to_array2_f32, array2_to_vec};
