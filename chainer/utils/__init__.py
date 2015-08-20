import numpy

from chainer.utils import walker_alias

WalkerAlias = walker_alias.WalkerAlias


def force_array(x):
    # numpy returns a float value (scalar) when a return value of an operator
    # is a 0-dimension array.
    # We need to convert such a value to a 0-dimension array because `Function`
    # object needs to return an `numpy.ndarray`.
    if numpy.isscalar(x):
        return numpy.array(x)
    else:
        return x


def force_type(dtype, value):
    if numpy.isscalar(value):
        return dtype.type(value)
    elif value.dtype != dtype:
        return value.astype(dtype)
    else:
        return value
