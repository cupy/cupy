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


bool_ = Scalar(numpy.bool_)


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
