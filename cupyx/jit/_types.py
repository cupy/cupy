import numpy
from cupy._core._scalar import get_typename


# Base class for cuda types.
class TypeBase:

    def __str__(self):
        raise NotImplementedError

    def declvar(self, x):
        return f'{self} {x}'


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

    def __init__(self, child_type, size):
        if not (isinstance(size, int) or size is None):
            raise 'size of shared_memory must be integer or `None`'
        self._size = size
        super().__init__(child_type, 1)

    def declvar(self, x):
        if self._size is None:
            return f'extern __shared__ {self.child_type} {x}[]'
        return f'__shared__ {self.child_type} {x}[{self._size}]'


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
        return self.types == other.types


void = Void()
bool_ = Scalar(numpy.bool_)
uint32 = Scalar(numpy.uint32)


_suffix_literals_dict = {
    numpy.dtype('float64'): '',
    numpy.dtype('float32'): 'f',
    numpy.dtype('int64'): 'll',
    numpy.dtype('longlong'): 'll',
    numpy.dtype('int32'): '',
    numpy.dtype('uint64'): 'ull',
    numpy.dtype('ulonglong'): 'ull',
    numpy.dtype('uint32'): 'u',
    numpy.dtype('bool'): '',
}


def get_cuda_code_from_constant(x, ctype):
    dtype = ctype.dtype
    suffix_literal = _suffix_literals_dict.get(dtype)
    if suffix_literal is not None:
        s = str(x).lower()
        return f'{s}{suffix_literal}'
    ctype = str(ctype)
    if dtype.kind == 'c':
        return f'{ctype}({x.real}, {x.imag})'
    if ' ' in ctype:
        return f'({ctype}){x}'
    return f'{ctype}({x})'
