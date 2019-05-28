import cupy
from cupy import core
from cupy.core import fusion


def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the maximum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis ``axis`` is preserved as an axis
            of length one.

    Returns:
        cupy.ndarray: The indices of the maximum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmax`

    """
    # TODO(okuta): check type
    return a.argmax(axis=axis, dtype=dtype, out=out, keepdims=keepdims)



def nanargmax(a, axis=None):
    """Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slice ``ValueError`` is raised.
    
    Subclass cannot be passed yet, subok=True still unsupported

    Args:
        a (cupy.ndarray): Array to take nanargmax.
        axis (int): Along which axis to find the maximum. ``a`` is flattened by
            default.
    Returns:
        cupy.ndarray: The indices of the maximum of ``a`` along an axis ignoring NaN values.
    .. seealso:: :func:`numpy.nanargmax`
    
    """
    a, mask = _replace_nan(a, -cupy.inf)
    res = argmax(a, axis=axis)

    if mask is not None:
        mask = cupy.all(mask, axis=axis)
        if cupy.any(mask):
            raise ValueError('All-NaN slice encountered')

    return res

def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the indices of the minimum along an axis.

    Args:
        a (cupy.ndarray): Array to take argmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis ``axis`` is preserved as an axis
            of length one.

    Returns:
        cupy.ndarray: The indices of the minimum of ``a`` along an axis.

    .. seealso:: :func:`numpy.argmin`

    """
    # TODO(okuta): check type
    return a.argmin(axis=axis, dtype=dtype, out=out, keepdims=keepdims)



def nanargmin(a, axis=None):
    """Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slice ``ValueError`` is raised.
    
    Subclass cannot be passed yet, subok=True still unsupported

    Args:
        a (cupy.ndarray): Array to take nanargmin.
        axis (int): Along which axis to find the minimum. ``a`` is flattened by
            default.
    Returns:
        cupy.ndarray: The indices of the minimum of ``a`` along an axis ignoring NaN values.
    .. seealso:: :func:`numpy.nanargmin`
    """
    a, mask = _replace_nan(a, cupy.inf)
    res = argmin(a, axis=axis)

    if mask is not None:
        mask = cupy.all(mask, axis=axis)
        if cupy.any(mask):
            raise ValueError('All-NaN slice encountered')

    return res

def _replace_nan(a, val):
        """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.
    Note that scalars will end up as array scalars, which is important
    for using the result as the value of the out argument in some
    operations.
    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.
    Returns
    -------
    y : cupy.ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.
    """

    a = cp.array(a, copy=True)
    ## dtype checking of a with cp.object_ not supported yet.
    if issubclass(a.dtype.type, cp.inexact):
        mask = cp.isnan(a)
    else:
        mask = None

    if mask is not None:
        cp.copyto(a, val, where=mask)

    return a, mask

# TODO(okuta): Implement argwhere


def nonzero(a):
    """Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of a,
    containing the indices of the non-zero elements in that dimension.

    Args:
        a (cupy.ndarray): array

    Returns:
        tuple of arrays: Indices of elements that are non-zero.

    .. seealso:: :func:`numpy.nonzero`

    """
    assert isinstance(a, core.ndarray)
    return a.nonzero()


def flatnonzero(a):
    """Return indices that are non-zero in the flattened version of a.

    This is equivalent to a.ravel().nonzero()[0].

    Args:
        a (cupy.ndarray): input array

    Returns:
        cupy.ndarray: Output array,
        containing the indices of the elements of a.ravel() that are non-zero.

    .. seealso:: :func:`numpy.flatnonzero`
    """
    assert isinstance(a, core.ndarray)
    return a.ravel().nonzero()[0]


_where_ufunc = core.create_ufunc(
    'cupy_where',
    ('???->?', '?bb->b', '?BB->B', '?hh->h', '?HH->H', '?ii->i', '?II->I',
     '?ll->l', '?LL->L', '?qq->q', '?QQ->Q', '?ee->e', '?ff->f',
     # On CUDA 6.5 these combinations don't work correctly (on CUDA >=7.0, it
     # works).
     # See issue #551.
     '?hd->d', '?Hd->d',
     '?dd->d', '?FF->F', '?DD->D'),
    'out0 = in0 ? in1 : in2')


def where(condition, x=None, y=None):
    """Return elements, either from x or y, depending on condition.

    If only condition is given, return ``condition.nonzero()``.

    Args:
        condition (cupy.ndarray): When True, take x, otherwise take y.
        x (cupy.ndarray): Values from which to choose on ``True``.
        y (cupy.ndarray): Values from which to choose on ``False``.

    Returns:
        cupy.ndarray: Each element of output contains elements of ``x`` when
            ``condition`` is ``True``, otherwise elements of ``y``. If only
            ``condition`` is given, return the tuple ``condition.nonzero()``,
            the indices where ``condition`` is True.

    .. seealso:: :func:`numpy.where`

    """

    missing = (x is None, y is None).count(True)

    if missing == 1:
        raise ValueError('Must provide both \'x\' and \'y\' or neither.')
    if missing == 2:
        return nonzero(condition)

    if fusion._is_fusing():
        return fusion._call_ufunc(_where_ufunc, condition, x, y)
    return _where_ufunc(condition.astype('?'), x, y)


# TODO(okuta): Implement searchsorted


# TODO(okuta): Implement extract
