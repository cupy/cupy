import numpy

import cupy


# TODO(asi1024): Duplicate of _scalar.pyx
_typenames_base = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}


class Void:

    def __init__(self):
        pass

    @property
    def ctype(self):
        return 'void'


class Scalar:

    def __init__(self, dtype):
        self.dtype = numpy.dtype(dtype)

    @property
    def ctype(self):
        return _typenames_base[self.dtype]


class Array:

    def __init__(self, dtype, ndim, c_contiguous, use_32bit_indexing):
        self.dtype = numpy.dtype(dtype)
        self.ndim = ndim
        self.c_contiguous = c_contiguous
        self.use_32bit_indexing = use_32bit_indexing

    @property
    def ctype(self):
        return 'CArray<{}, {}, {}, {}>'.format(
            _typenames_base[self.dtype],
            self.ndim,
            str(self.c_contiguous).lower(),
            str(self.use_32bit_indexing).lower())


# Currently not used
class Indexer:

    def __init__(self, ndim):
        self.ndim = ndim

    @property
    def ctype(self):
        return 'CIndexer<{}>'.format(self.ndim)


# Currently not used
class Pointer:

    def __init__(self, t):
        self.t = t
        self.dtype = t.dtype

    @property
    def ctype(self):
        self.ctype = '{}*'.format(t.ctype)


def _type_from_obj(x):
    if isinstance(x, int):
        return Scalar(numpy.int32)
    if isinstance(x, float):
        return Scalar(numpy.flaot32)
    if isinstance(x, complex):
        return Scalar(numpy.complex64)
    if isinstance(x, numpy.generic):
        return Scalar(x.dtype)
    if isinstance(x, cupy.ndarray):
        return Array(x.dtype, x.ndim, x._c_contiguous, x._index_32_bits)
    raise ValueError('Unsupported type: {}'.format(type(x)))
