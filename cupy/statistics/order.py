import warnings

from cupy import core
from cupy.logic import content


def amin(a, axis=None, out=None, keepdims=False, dtype=None):
    """Returns the minimum of an array or the minimum along an axis.

    .. note::

       When at least one element is NaN, the corresponding min value will be
       NaN.

    Args:
        a (cupy.ndarray): Array to take the minimum.
        axis (int): Along which axis to take the minimum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.amin`

    """
    # TODO(okuta): check type
    return a.min(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def amax(a, axis=None, out=None, keepdims=False, dtype=None):
    """Returns the maximum of an array or the maximum along an axis.

    .. note::

       When at least one element is NaN, the corresponding min value will be
       NaN.

    Args:
        a (cupy.ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        dtype: Data type specifier.

    Returns:
        cupy.ndarray: The maximum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.amax`

    """
    # TODO(okuta): check type
    return a.max(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanmin(a, axis=None, out=None, keepdims=False):
    """Returns the minimum of an array along an axis ignoring NaN.

    When there is a slice whose elements are all NaN, a :class:`RuntimeWarning`
    is raised and NaN is returned.

    Args:
        a (cupy.ndarray): Array to take the minimum.
        axis (int): Along which axis to take the minimum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.nanmin`

    """
    res = core.nanmin(a, axis=axis, out=out, keepdims=keepdims)
    if content.isnan(res).any():
        warnings.warn('All-NaN slice encountered', RuntimeWarning)
    return res


def nanmax(a, axis=None, out=None, keepdims=False):
    """Returns the maximum of an array along an axis ignoring NaN.

    When there is a slice whose elements are all NaN, a :class:`RuntimeWarning`
    is raised and NaN is returned.

    Args:
        a (cupy.ndarray): Array to take the maximum.
        axis (int): Along which axis to take the maximum. The flattened array
            is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The maximum of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.nanmax`

    """
    res = core.nanmax(a, axis=axis, out=out, keepdims=keepdims)
    if content.isnan(res).any():
        warnings.warn('All-NaN slice encountered', RuntimeWarning)
    return res


# TODO(okuta): Implement ptp


# TODO(okuta): Implement percentile
