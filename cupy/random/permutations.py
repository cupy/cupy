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
    """Returns a permuted range or shuffles an array."""
    if isinstance(a, six.integer_types):
        rs = generator.get_random_state()
        return rs.permutation(a)
    else:
        return shuffle(a)
