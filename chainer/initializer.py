import numpy


class Initializer(object):

    dtype = numpy.float32

    def __init__(self, dtype=None):
        if dtype is not None:
            self.dtype = dtype

    def generate_array(self, shape, xp):
        """Return initialized array.

        The algorithms used to make the new values depend on the
        concrete derived classes.

        Args:
            shape (tuple): Shape of a return array.
            xp (module): :mod:`cupy` or :mod:`numpy`.

        Returns:
            numpy.ndarray or cupy.ndarray: An initialized array.

        """
        array = xp.empty(shape, dtype=self.dtype)
        self(array)
        return array

    def __call__(self, array):
        """Initializes given array.

        This method destructively changes the value of array.
        The derived class is required to implement this method.
        The algorithms used to make the new values depend on the
        concrete derived classes.

        Args:
            array (numpy.ndarray or cupy.ndarray):
                An array to be initialized by this initializer.

        Returns:
            numpy.ndarray or cupy.ndarray: An initialized array.

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
