import cupy
from cupy.random import generator

# TODO(okuta): Implement permutation


def shuffle(a):
    """Shuffles an array.

    Args:
        a (cupy.ndarray): The array to be shuffled.

    .. seealso:: :func:`numpy.random.shuffle`

    """
    if not isinstance(a, cupy.ndarray):
        raise TypeError('The array must be cupy.ndarray')

    if a.ndim == 0:
        raise TypeError('An array whose ndim is 0 is not supported')

    rs = generator.get_random_state()
    return rs.shuffle(a)
