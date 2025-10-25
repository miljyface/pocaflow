mod validation;

pub use validation::validate_matmul_dims;
pub use validation::validate_vector_lengths;
pub use validation::validate_3d_vectors;
pub use validation::validate_nonzero_magnitude;
pub use validation::validate_nonzero_magnitude_f32;
