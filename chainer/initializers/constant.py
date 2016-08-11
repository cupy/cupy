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

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
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

    def __init__(self, fill_value):
        self.fill_value = fill_value

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        array[...] = xp.asarray(self.fill_value)


def Zero():

    """Returns initializer that initializes array with the all-zero array."""

    return Constant(0.0)


def One():

    """Returns initializer that initializes array with the all-one array."""

    return Constant(1.0)
