import numpy

from chainer import cuda


def _make_ordinal(n):
    ORDINAL_SUFFIX = {1: 'st', 2: 'nd', 3: 'rd'}
    if n % 100 < 10:
        i = n % 10
    else:
        i = n

    suffix = ORDINAL_SUFFIX.get(i, 'th')
    return '{0}{1}'.format(n, suffix)


class TypeInfo(object):
    def __init__(self, index, shape, dtype):
        self.index = index
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def ordinal(self):
        return _make_ordinal(self.index + 1)


def get_type(index, array):
    assert(isinstance(array, numpy.ndarray) or
           isinstance(array, cuda.GPUArray))
    return TypeInfo(index, array.shape, array.dtype)


class InvalidType(Exception):
    pass


class InvalidArgumentLength(InvalidType):
    pass


class InvalidArgumentType(InvalidType):
    pass


def assert_argument_size_equals(types, expect):
    assert(isinstance(types, tuple))

    if len(types) != expect:
        msg = '# of arguments is epxected to be {1}, but is {2}'.format(
            expect, len(types))
        raise InvalidArgumentLength(msg)


def assert_type_equals(type_info, expect):
    assert(isinstance(type_info, TypeInfo))

    if type_info.dtype.kind != expect:
        msg = 'Type of {0} argument is expected to be {1}, but is {2}'.format(
            type_info.ordinal(), expect, type_info.dtype.kind)
        raise InvalidArgumentType(msg)


def assert_ndim_equals(type_info, expect):
    assert(isinstance(type_info, TypeInfo))

    if type_info.ndim != expect:
        msg = '# of dimensions of {0} argument is expected to be {1}, but is {2}'.format(
            type_info.ordinal(), expect, type_info.ndim)
        raise InvalidArgumentType(msg)


def assert_shape_equals(type_info, index, expect):
    assert(isinstance(type_info, TypeInfo))

    dim = type_info.shape[index]
    if dim != expect:
        msg = '{0} dimension of {1} argument is expected to be {2}, but is {3}'.format(
            _make_ordinal(index + 1), type_info.ordinal(), expect, dim)
        raise InvalidArgumentType(msg)
