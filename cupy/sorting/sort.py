def sort(a):
    """Returns a sorted copy of an array with a stable sorting algorithm.

    Args:
        a (cupy.ndarray): Array to be sorted.

    Returns:
        cupy.ndarray: Array of the same type and shape as ``a``.

    .. note::
       For its implementation reason, ``cupy.sort`` currently supports only
       arrays with their rank of one and does not support ``axis``, ``kind``
       and ``order`` parameters that ``numpy.sort`` does support.

    .. seealso:: :func:`numpy.sort`

    """
    ret = a.copy()
    ret.sort()
    return ret


# TODO(okuta): Implement lexsort


# TODO(okuta): Implement argsort


# TODO(okuta): Implement msort


# TODO(okuta): Implement sort_complex


# TODO(okuta): Implement partition


# TODO(okuta): Implement argpartition
