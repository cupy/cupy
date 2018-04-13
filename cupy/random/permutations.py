from cupy.random import generator
import six


def shuffle(a):
    """Shuffles an array.

    Args:
        a (cupy.ndarray): The array to be shuffled.

    .. seealso:: :func:`numpy.random.shuffle`

    """
    rs = generator.get_random_state()
    return rs.shuffle(a)


def permutation(a):
    """Returns a permuted range or a permutation of an array.

    Args:
        a (int or cupy.ndarray): The range or the array to be shuffled.

    Returns:
        cupy.ndarray: If `a` is an integer, it is permutation range between 0
        and `a` - 1.
        Otherwise, it is a permutation of `a`.

    .. seealso:: :func:`numpy.random.permutation`
    """
    rs = generator.get_random_state()
    if isinstance(a, six.integer_types):
        return rs.permutation(a)
    else:
        return a[rs.permutation(len(a))]
