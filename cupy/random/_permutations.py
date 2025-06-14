from cupy.random import _generator
import numpy as np

def shuffle(a):
    """Shuffles an array.

    Args:
        a (cupy.ndarray): The array to be shuffled.

    .. seealso:: :meth:`numpy.random.shuffle`

    """
    rs = _generator.get_random_state()
    return rs.shuffle(a)


def permutation(a):
    """Returns a permuted range or a permutation of an array.

    Args:
        a (int or cupy.ndarray): The range or the array to be shuffled.

    Returns:
        cupy.ndarray: If `a` is an integer, it is permutation range between 0
        and `a` - 1.
        Otherwise, it is a permutation of `a`.

    .. seealso:: :meth:`numpy.random.permutation`
    """
    rs = _generator.get_random_state()
    return rs.permutation(a)


def permuted(a, axis=0,type=np.uint16):
    """Returns a permutation of an array for each row.

    Args:
        a (int or cupy.ndarray): The range or the array to be shuffled.
        axis (int >= 0): the axis that will be shuffled

    Returns:
        cupy.ndarray: If `a` is an integer, it is permutation range between 0
        and `a` - 1.
        Otherwise, it is a permutation of `a`.

    .. seealso:: :meth:`numpy.random.permuted`
    """
    rs = _generator.get_random_state()
    return rs.permuted(a, axis=axis, type=type)
