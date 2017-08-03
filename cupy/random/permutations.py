import numpy

import cupy

# TODO(okuta): Implement permutation


def shuffle(a):
    """Shuffles an array.

    Args:
        a (cupy.ndarray): The array to be shuffled.

    .. seealso:: :func:`numpy.random.shuffle`

    """
    if not isinstance(a, cupy.ndarray):
        msg = 'a must be cupy.ndarray'
        raise TypeError(msg)

    if a.ndim == 0:
        msg = 'An array whose ndim is 0 is not supported'
        raise TypeError(msg)

    int_max = numpy.iinfo(numpy.int32).max
    int_min = numpy.iinfo(numpy.int32).min
    a[:] = a[cupy.argsort(cupy.random.randint(int_min, int_max, size=len(a)))]
