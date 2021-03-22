import numpy
from cupy.core._scalar import get_typename


# Base class for cuda types.
class TypeBase:
    pass


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
        return self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype)


class Array(TypeBase):

    def __init__(self, dtype, ndim, is_c_contiguous, index_32_bits):
        self.dtype = dtype
        self.ndim = ndim
        self.is_c_contiguous = is_c_contiguous
        self.index_32_bits = index_32_bits

    @classmethod
    def from_ndarray(cls, x):
        return Array(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)

    def __str__(self):
        ctype = get_typename(self.dtype)
        c_contiguous = get_cuda_code_from_constant(self.is_c_contiguous, bool_)
        index_32_bits = get_cuda_code_from_constant(self.index_32_bits, bool_)
        return f'CArray<{ctype}, {self.ndim}, {c_contiguous}, {index_32_bits}>'

    def __eq__(self, other):
        return (
            self.dtype == other.dtype and
            self.ndim == other.ndim and
            self.is_c_contiguous == other.is_c_contiguous and
            self.index_32_bits == other.index_32_bits
        )

    def __hash__(self):
        return hash(
            (self.dtype, self.ndim, self.is_c_contiguous, self.index_32_bits))


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
