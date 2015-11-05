# TODO(okuta): Implement tile

import collections
import six

import cupy


def repeat(a, repeats, axis=None):
    """Repeat arrays along an axis.

    Args:
        a (cupy.ndarray): Array to transform.
        repeats (int, list or tuple): The number of repeats.
        axis (int): The axis to repeat.

    Returns:
        cupy.ndarray: Transformed array with repeats.

    .. seealso:: :func:`numpy.repeat`

    """
    if isinstance(repeats, int):
        if repeats < 0:
            raise ValueError(
                "'repeats' should not be negative: {}".format(repeats))
        if axis is None:
            b = a.reshape((-1, 1))
            ret = cupy.empty((len(b), repeats), dtype=b.dtype)
            if ret.size:
                ret[...] = b
            return ret.ravel()

        repeats = [repeats] * a.shape[axis]
    elif isinstance(repeats, collections.Sequence):
        if any(rep < 0 for rep in repeats):
            raise ValueError(
                "all elements of 'repeats' should not be negative: {}"
                .format(repeats))
        if axis is None:
            raise ValueError(
                "'axis' should be specified if 'repeats' is sequence")
        if a.shape[axis] != len(repeats):
            raise ValueError(
                "'repeats' and 'axis' of 'a' should be same length: {} != {}"
                .format(a.shape[axis], len(repeats)))
    else:
        raise ValueError(
            "'repeats' should be int or sequence: {}".format(repeats))

    if axis < 0:
        axis = len(a.shape) + axis

    ret_shape = list(a.shape)
    ret_shape[axis] = sum(repeats)
    ret = cupy.empty((ret_shape), dtype=a.dtype)
    a_index = [slice(None)] * len(ret_shape)
    ret_index = list(a_index)
    offset = 0
    for i in six.moves.xrange(a.shape[axis]):
        if repeats[i] == 0:
            continue
        a_index[axis] = slice(i, i + 1)
        ret_index[axis] = slice(offset, offset + repeats[i])
        # convert to tuple because cupy has a indexing bug
        ret[tuple(ret_index)] = a[tuple(a_index)]
        offset += repeats[i]
    return ret
