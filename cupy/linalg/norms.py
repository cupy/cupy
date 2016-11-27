import numpy

import cupy


def norm(x, ord=None, axis=None, keepdims=False):
    """Returns one of matrix norms specified by ``ord`` parameter.

    Complex valued matrices and vectors are not supported.
    See numpy.linalg.norm for more detail.

    Args:
        x (cupy.ndarray): Array to take norm. If ``axis`` is None,
            ``x`` must be 1-D or 2-D.
        ord (non-zero int, inf, -inf, 'fro'): Norm type.
        axis (int, 2-tuple of ints, None): 1-D or 2-D norm is cumputed over
            ``axis``.
        keepdims (bool): If this is set ``True``, the axes which are normed
            over are left.

    Returns:
        cupy.ndarray

    """
    if not issubclass(x.dtype.type, numpy.inexact):
        x = x.astype(float)

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if (ord is None or (ndim == 1 and ord == 2) or
                (ndim == 2 and ord in ('f', 'fro'))):
            ret = cupy.sqrt(cupy.sum(x.ravel() ** 2))
            if keepdims:
                ret = ret.reshape((1,) * ndim)
            return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception:
            raise TypeError(
                "'axis' must be None, an integer or a tuple of integers")
        axis = (axis,)

    if len(axis) == 1:
        if ord == numpy.Inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -numpy.Inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            # Convert to Python float in accordance with NumPy
            return (x != 0).sum(axis=axis, keepdims=keepdims, dtype='d')
        elif ord == 1:
            # special case for speedup
            return abs(x).sum(axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = x ** 2
            return cupy.sqrt(s.sum(axis=axis, keepdims=keepdims))
        else:
            try:
                float(ord)
            except TypeError:
                raise ValueError("Invalid norm order for vectors.")
            absx = abs(x).astype('d')
            absx **= ord
            ret = absx.sum(axis=axis, keepdims=keepdims)
            ret **= (1.0 / ord)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        if row_axis < 0:
            row_axis += nd
        if col_axis < 0:
            col_axis += nd
        if not (0 <= row_axis < nd and 0 <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).max(axis=col_axis)
        elif ord == numpy.Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).min(axis=col_axis)
        elif ord == -numpy.Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = cupy.sqrt((x ** 2).sum(axis=axis))
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")


# TODO(okuta): Implement cond


# TODO(okuta): Implement det


# TODO(okuta): Implement matrix_rank


# TODO(okuta): Implement slogdet


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Returns the sum along the diagonals of an array.

    It computes the sum along the diagonals at ``axis1`` and ``axis2``.

    Args:
        a (cupy.ndarray): Array to take trace.
        offset (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.
        axis1 (int): The first axis along which the trace is taken.
        axis2 (int): The second axis along which the trace is taken.
        dtype: Data type specifier of the output.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The trace of ``a`` along axes ``(axis1, axis2)``.

    .. seealso:: :func:`numpy.trace`

    """
    # TODO(okuta): check type
    return a.trace(offset, axis1, axis2, dtype, out)
