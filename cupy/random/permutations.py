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
    rs = generator.get_random_state()
    if isinstance(a, six.integer_types):
        return rs.permutation(a)
    else:
        return a[rs.permutation(len(a))]
