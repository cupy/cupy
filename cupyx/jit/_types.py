import numpy


_typenames = {
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
        return _typenames[dtype]
