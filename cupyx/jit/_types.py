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
