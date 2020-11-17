from cupy.random import _random_state


def shuffle(a):
    """Shuffles an array.

    Args:
        a (cupy.ndarray): The array to be shuffled.

    .. seealso:: :meth:`numpy.random.shuffle`

    """
    rs = _random_state.get_random_state()
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
    rs = _random_state.get_random_state()
    return rs.permutation(a)
