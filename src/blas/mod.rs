mod bindings;
mod operations;

pub use bindings::{CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, cblas_dgemm, cblas_sgemm, cblas_daxpy, cblas_saxpy};
pub use operations::{dgemm, sgemm};
