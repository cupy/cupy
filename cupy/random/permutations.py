from cupy.random import generator

# TODO(okuta): Implement permutation


def shuffle(a):
    """Shuffles an array.

    Args:
        a (cupy.ndarray): The array to be shuffled.

    .. seealso:: :func:`numpy.random.shuffle`

    """
    rs = generator.get_random_state()
    return rs.shuffle(a)
