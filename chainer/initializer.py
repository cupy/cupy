import numpy


class Initializer(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, array=None, shape=None, xp=None):
        """Initializes given array.

        This method destructively changes the value of array.
        The derived class is required to implement this method.
        The algorithms used to make the new values depend on the
        concrete derived classes.

        Args:
            array (numpy.ndarray or cupy.ndarray):
                An array to be initialized by this initializer. If ``None``,
                this method returns new array.
            shape (tuple): Shape of a return array. If ``None``, this method
                uses ``array.shape``.
            xp (module): :mod:`cupy` or :mod:`numpy`. If ``None``, this method
                uses ``cuda.get_array_module(array)``.

        """
        raise NotImplementedError()


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

def get_fans(shape):
    if not isinstance(shape, tuple):
        raise ValueError('shape must be tuple')

    if len(shape) < 2:
        raise ValueError('shape must be of length >= 2: shape={}', shape)

    fan_in = numpy.prod(shape[1:])
    fan_out = shape[0]
    return fan_in, fan_out
