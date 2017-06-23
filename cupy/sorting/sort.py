import cupy
import numpy

try:
    from cupy.cuda import thrust
except ImportError:
    pass


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


def lexsort(keys):
    """Perform an indirect sort using an array of keys.

    Args:
        keys (cupy.ndarray): ``(k, N)`` array containing ``k`` ``(N,)``-shaped
            arrays. The ``k`` different "rows" to be sorted. The last row is
            the primary sort key.

    Returns:
        cupy.ndarray: Array of indices that sort the keys.

    .. note::
        For its implementation reason, ``cupy.lexsort`` currently supports only
        keys with their rank of one or two and does not support ``axis``
        parameter that ``numpy.lexsort`` supports.

    .. seealso:: :func:`numpy.lexsort`

    """

    # TODO(takagi): Support axis argument.

    if keys.ndim == ():
        # as numpy.lexsort() raises
        raise TypeError('need sequence of keys with len > 0 in lexsort')

    if keys.ndim == 1:
        return 0

    # TODO(takagi): Support ranks of three or more.
    if keys.ndim > 2:
        raise NotImplementedError('Keys with the rank of three or more is not '
                                  'supported in lexsort')

    idx_array = cupy.ndarray(keys._shape[1:], dtype=numpy.intp)
    k = keys._shape[0]
    n = keys._shape[1]

    try:
        thrust.lexsort(keys.dtype, idx_array.data.ptr, keys.data.ptr, k, n)
    except NameError:
        raise RuntimeError('Thrust is needed to use cupy.lexsort. Please '
                           'install CUDA Toolkit with Thrust then reinstall '
                           'CuPy after uninstalling it.')

    return idx_array


def argsort(a):
    """Return the indices that would sort an array with a stable sorting.

    Args:
        a (cupy.ndarray): Array to sort.

    Returns:
        cupy.ndarray: Array of indices that sort ``a``.

    .. note::
       For its implementation reason, ``cupy.argsort`` currently supports only
       arrays with their rank of one and does not support ``axis``, ``kind``
       and ``order`` parameters that ``numpy.argsort`` supports.

    .. seealso:: :func:`numpy.argsort`

    """
    return a.argsort()


# TODO(okuta): Implement msort


# TODO(okuta): Implement sort_complex


# TODO(okuta): Implement partition


# TODO(okuta): Implement argpartition
