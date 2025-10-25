from typing import List, Union, Tuple
import random

class tensor:
    def __init__(self, data: List, dtype: str = 'float64'):
        if dtype not in ['float64', 'float32']:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        self._dtype = dtype
        self._data = self._convert_dtype(data, dtype)
        self._shape = self._infer_shape(self._data)
        self._ndim = len(self._shape)
    
    @staticmethod
    def _convert_dtype(data, dtype):
        """Convert all numbers to specified dtype"""
        cast = float
        
        def convert_recursive(d):
            if isinstance(d, list):
                return [convert_recursive(x) for x in d]
            return cast(d)
        
        return convert_recursive(data)
    
    @staticmethod
    def _infer_shape(data) -> Tuple[int, ...]:
        """Infer shape from nested lists"""
        shape = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            if len(current) > 0:
                current = current[0]
            else:
                break
        return tuple(shape)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def ndim(self) -> int:
        return self._ndim
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def data(self) -> List:
        return self._data

    def __iter__(self):
        """Make Tensor iterable like a list"""
        return iter(self._data)
    
    def __len__(self):
        """Support len() operation"""
        if self.ndim == 0:
            return 0
        return self._shape[0] if self._shape else 0
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})\n{self._data}"
    
    def __str__(self) -> str:
        return self._format_data(self._data)
    
    def _format_data(self, data, indent=0) -> str:
        if not isinstance(data, list):
            return str(data)
        
        if all(not isinstance(x, list) for x in data):
            return '[' + ', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in data) + ']'
        
        lines = ['[']
        for i, item in enumerate(data):
            prefix = '  ' * (indent + 1)
            lines.append(prefix + self._format_data(item, indent + 1) + (',' if i < len(data) - 1 else ''))
        lines.append('  ' * indent + ']')
        return '\n'.join(lines)
    
    def __getitem__(self, key):
        """Support indexing"""
        result = self._data[key]
        if isinstance(result, list):
            return Tensor(result, dtype=self._dtype)
        return result
    
    def tolist(self) -> List:
        """Convert to Python list"""
        return self._data
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: str = 'float64') -> 'Tensor':
        """Create tensor filled with zeros"""
        def create_nested(dims):
            if len(dims) == 1:
                return [0.0] * dims[0]
            return [create_nested(dims[1:]) for _ in range(dims[0])]
        
        return Tensor(create_nested(shape), dtype=dtype)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: str = 'float64') -> 'Tensor':
        """Create tensor filled with ones"""
        def create_nested(dims):
            if len(dims) == 1:
                return [1.0] * dims[0]
            return [create_nested(dims[1:]) for _ in range(dims[0])]
        
        return Tensor(create_nested(shape), dtype=dtype)
    
    @staticmethod
    def random(shape: Tuple[int, ...], dtype: str = 'float64') -> 'Tensor':
        """Create tensor with random values [0, 1)"""
        def create_nested(dims):
            if len(dims) == 1:
                return [random.random() for _ in range(dims[0])]
            return [create_nested(dims[1:]) for _ in range(dims[0])]
        
        return Tensor(create_nested(shape), dtype=dtype)
    
    @staticmethod
    def eye(n: int, dtype: str = 'float64') -> 'Tensor':
        """Create identity matrix"""
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return Tensor(data, dtype=dtype)