use std::mem;

/// Data type enum for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U8,
    F16,
}

impl DType {
    /// Size in bytes
    #[inline]
    pub const fn size(&self) -> usize {
        match self {
            DType::F32 => mem::size_of::<f32>(),
            DType::F64 => mem::size_of::<f64>(),
            DType::I32 => mem::size_of::<i32>(),
            DType::I64 => mem::size_of::<i64>(),
            DType::U8 => mem::size_of::<u8>(),
            DType::F16 => 2,
        }
    }

    /// Alignment in bytes
    #[inline]
    pub const fn alignment(&self) -> usize {
        match self {
            DType::F32 => mem::align_of::<f32>(),
            DType::F64 => mem::align_of::<f64>(),
            DType::I32 => mem::align_of::<i32>(),
            DType::I64 => mem::align_of::<i64>(),
            DType::U8 => mem::align_of::<u8>(),
            DType::F16 => 2,
        }
    }

    #[inline]
    pub const fn is_floating_point(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16)
    }

    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8)
    }

    pub const fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::F16 => "f16",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}