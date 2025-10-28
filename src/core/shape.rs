/// Tensor shape with compile-time rank checking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    #[inline]
    pub fn new(dims: Vec<usize>) -> Self {
        assert!(!dims.is_empty(), "Shape cannot be empty");
        Self { dims }
    }

    #[inline]
    pub fn scalar() -> Self {
        Self { dims: vec![1] }
    }

    #[inline]
    pub fn from_slice(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        // For now, assume all shapes are contiguous
        true
    }

    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let max_rank = self.rank().max(other.rank());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let a_dim = if i < self.rank() {
                self.dims[self.rank() - 1 - i]
            } else {
                1
            };
            let b_dim = if i < other.rank() {
                other.dims[other.rank() - 1 - i]
            } else {
                1
            };

            if a_dim == b_dim {
                result.push(a_dim);
            } else if a_dim == 1 {
                result.push(b_dim);
            } else if b_dim == 1 {
                result.push(a_dim);
            } else {
                return None;
            }
        }

        result.reverse();
        Some(Shape::new(result))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::from_slice(dims)
    }
}

/// Stride information for tensor layout
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stride {
    strides: Vec<usize>,
}

impl Stride {
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![1; dims.len()];
        
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        Self { strides }
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self, indices: &[usize]) -> usize {
        indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_numel() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::new(vec![3, 1, 5]);
        let b = Shape::new(vec![1, 4, 5]);
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[3, 4, 5]);
    }

    #[test]
    fn test_stride() {
        let shape = Shape::new(vec![2, 3, 4]);
        let stride = Stride::contiguous(&shape);
        assert_eq!(stride.strides(), &[12, 4, 1]);
    }
}