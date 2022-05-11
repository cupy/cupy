import numpy
from cupy._core._scalar import get_typename


# Base class for cuda types.
class TypeBase:

    def __str__(self):
        raise NotImplementedError

    def declvar(self, x, init):
        if init is None:
            return f'{self} {x}'
        return f'{self} {x} = {init.code}'

    def assign(self, var, value):
        return f'{var.code} = {value.code}'


class Void(TypeBase):

    def __init__(self):
        pass

    def __str__(self):
        return 'void'


class Scalar(TypeBase):

    def __init__(self, dtype):
        self.dtype = numpy.dtype(dtype)

    def __str__(self):
        dtype = self.dtype
        if dtype == numpy.float16:
            # For the performance
            dtype = numpy.dtype('float32')
        return get_typename(dtype)

    def __eq__(self, other):
        return isinstance(other, Scalar) and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype)


class PtrDiff(Scalar):
    def __init__(self):
        super().__init__('q')

    def __str__(self):
        return 'ptrdiff_t'


class ArrayBase(TypeBase):

    def __init__(self, child_type: TypeBase, ndim: int):
        assert isinstance(child_type, TypeBase)
        self.child_type = child_type
        self.ndim = ndim


class CArray(ArrayBase):
    def __init__(self, dtype, ndim, is_c_contiguous, index_32_bits):
        self.dtype = dtype
        self._c_contiguous = is_c_contiguous
        self._index_32_bits = index_32_bits
        super().__init__(Scalar(dtype), ndim)

    @classmethod
    def from_ndarray(cls, x):
        return CArray(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)

    def __str__(self):
        ctype = get_typename(self.dtype)
        c_contiguous = get_cuda_code_from_constant(self._c_contiguous, bool_)
        index_32_bits = get_cuda_code_from_constant(self._index_32_bits, bool_)
        return f'CArray<{ctype}, {self.ndim}, {c_contiguous}, {index_32_bits}>'

    def __eq__(self, other):
        return (
            isinstance(other, CArray) and
            self.dtype == other.dtype and
            self.ndim == other.ndim and
            self._c_contiguous == other._c_contiguous and
            self._index_32_bits == other._index_32_bits
        )

    def __hash__(self):
        return hash(
            (self.dtype, self.ndim, self._c_contiguous, self._index_32_bits))


class SharedMem(ArrayBase):

    def __init__(self, child_type, size, alignment=None):
        if not (isinstance(size, int) or size is None):
            raise 'size of shared_memory must be integer or `None`'
        if not (isinstance(alignment, int) or alignment is None):
            raise 'alignment must be integer or `None`'
        self._size = size
        self._alignment = alignment
        super().__init__(child_type, 1)

    def declvar(self, x, init):
        assert init is None
        if self._alignment is not None:
            code = f'__align__({self._alignment})'
        else:
            code = ''
        if self._size is None:
            code = f'extern {code} __shared__ {self.child_type} {x}[]'
        else:
            code = f'{code} __shared__ {self.child_type} {x}[{self._size}]'
        return code


class Ptr(ArrayBase):

    def __init__(self, child_type):
        super().__init__(child_type, 1)

    def __str__(self):
        return f'{self.child_type}*'


class Tuple(TypeBase):

    def __init__(self, types):
        self.types = types

    def __str__(self):
        types = ', '.join([str(t) for t in self.types])
        return f'thrust::tuple<{types}>'

    def __eq__(self, other):
        return isinstance(other, Tuple) and self.types == other.types


void = Void()
bool_ = Scalar(numpy.bool_)
int32 = Scalar(numpy.int32)
uint32 = Scalar(numpy.uint32)
uint64 = Scalar(numpy.uint64)


class Dim3(TypeBase):
    """
    An integer vector type based on uint3 that is used to specify dimensions.

    Attributes:
        x (uint32)
        y (uint32)
        z (uint32)
    """

    def x(self, code: str):
        from cupyx.jit import _internal_types  # avoid circular import
        return _internal_types.Data(f'{code}.x', uint32)

    def y(self, code: str):
        from cupyx.jit import _internal_types  # avoid circular import
        return _internal_types.Data(f'{code}.y', uint32)

    def z(self, code: str):
        from cupyx.jit import _internal_types  # avoid circular import
        return _internal_types.Data(f'{code}.z', uint32)

    def __str__(self):
        return 'dim3'


dim3 = Dim3()


_suffix_literals_dict = {
    'float64': '',
    'float32': 'f',
    'int64': 'll',
    'int32': '',
    'uint64': 'ull',
    'uint32': 'u',
    'bool': '',
}


def get_cuda_code_from_constant(x, ctype):
    dtype = ctype.dtype
    suffix_literal = _suffix_literals_dict.get(dtype.name)
    if suffix_literal is not None:
        s = str(x).lower()
        return f'{s}{suffix_literal}'
    ctype = str(ctype)
    if dtype.kind == 'c':
        return f'{ctype}({x.real}, {x.imag})'
    if ' ' in ctype:
        return f'({ctype}){x}'
    return f'{ctype}({x})'
