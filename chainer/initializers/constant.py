from chainer import cuda
from chainer import initializer


class Identity(initializer.Initializer):

    """Initializes array with the identity matrix.

    It initializes the given array with the constant
    multiple of the identity matrix.
    Note that arrays to be passed must be 2D squared matrices.

    Attributes:
        scale (scalar): A constant to be multiplied to identity matrices.

    """

    def __init__(self, scale=1.0, **kwargs):
        super(Identity, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self, array=None, shape=None, xp=None):
        if array is None:
            assert isinstance(shape, tuple)
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError('Identity matrix initialization can only be '
                                 'used for 2D squared matrices.')
            ret = xp.zeros(shape).astype(self.dtype)
            if xp is numpy:
                numpy.fill_diagonal(ret, self.scale)
            else:
                # TODO(okuta): Use fill_diagonal
                ret.diagonal()[...] = self.scale
            return ret
        assert self.dtype is None or array.dtype == self.dtype
        shape = array.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initialization can only be used '
                             'for 2D squared matrices.')
        array[...] = 0
        xp = cuda.get_array_module(array)
        xp.fill_diagonal(array, self.scale)


class Constant(initializer.Initializer):

    """Initializes array with constant value.

    Attributes:
        fill_value (scalar or numpy.ndarray or cupy.ndarray):
            A constant to be assigned to the initialized array.
            Broadcast is allowed on this assignment.
    """

    def __init__(self, fill_value, **kwargs):
        super(Constant, self).__init__(**kwargs)
        self.fill_value = fill_value

    def __call__(self, array=None, shape=None, xp=None):
        if array is None:
            assert isinstance(shape, tuple)
            return xp.full(shape, self.fill_value, self.dtype)
        assert self.dtype is None or array.dtype == self.dtype
        xp = cuda.get_array_module(array)
        array[...] = xp.asarray(self.fill_value)


def Zero(dtype=numpy.float32):
    """Returns initializer that initializes array with the all-zero array."""

    return Constant(0.0, dtype=dtype)


def One():
    """Returns initializer that initializes array with the all-one array."""

    return Constant(1.0, dtype=dtype)
