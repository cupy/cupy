import warnings

import cupy
from cupy import _core
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content


def amin(a, axis=None, out=None, keepdims=False):
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

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. note::
       When cuTENSOR accelerator is used, the output value might be collapsed
       for reduction axes that have one or more NaN elements.

    .. seealso:: :func:`numpy.amin`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.amin does not support `keepdims` in fusion yet.')
        return _fusion_thread_local.call_reduction(
            _statistics.amin, a, axis=axis, out=out)

    # TODO(okuta): check type
    return a.min(axis=axis, out=out, keepdims=keepdims)


def amax(a, axis=None, out=None, keepdims=False):
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

    Returns:
        cupy.ndarray: The maximum of ``a``, along the axis if specified.

    .. note::
       When cuTENSOR accelerator is used, the output value might be collapsed
       for reduction axes that have one or more NaN elements.

    .. seealso:: :func:`numpy.amax`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.amax does not support `keepdims` in fusion yet.')
        return _fusion_thread_local.call_reduction(
            _statistics.amax, a, axis=axis, out=out)

    # TODO(okuta): check type
    return a.max(axis=axis, out=out, keepdims=keepdims)


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

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.nanmin`

    """
    # TODO(niboshi): Avoid synchronization.
    res = _core.nanmin(a, axis=axis, out=out, keepdims=keepdims)
    if content.isnan(res).any():  # synchronize!
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

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.nanmax`

    """
    # TODO(niboshi): Avoid synchronization.
    res = _core.nanmax(a, axis=axis, out=out, keepdims=keepdims)
    if content.isnan(res).any():  # synchronize!
        warnings.warn('All-NaN slice encountered', RuntimeWarning)
    return res


def ptp(a, axis=None, out=None, keepdims=False):
    """Returns the range of values (maximum - minimum) along an axis.

    .. note::

       The name of the function comes from the acronym for 'peak to peak'.

       When at least one element is NaN, the corresponding ptp value will be
       NaN.

    Args:
        a (cupy.ndarray): Array over which to take the range.
        axis (int): Axis along which to take the minimum. The flattened
            array is used by default.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is retained as an axis of
            size one.

    Returns:
        cupy.ndarray: The minimum of ``a``, along the axis if specified.

    .. note::
       When cuTENSOR accelerator is used, the output value might be collapsed
       for reduction axes that have one or more NaN elements.

    .. seealso:: :func:`numpy.amin`

    """
    return a.ptp(axis=axis, out=out, keepdims=keepdims)


def _quantile_unchecked(a, q, axis=None, out=None, interpolation='linear',
                        keepdims=False):
    if q.ndim == 0:
        q = q[None]
        zerod = True
    else:
        zerod = False
    if q.ndim > 1:
        raise ValueError('Expected q to have a dimension of 1.\n'
                         'Actual: {0} != 1'.format(q.ndim))
    if keepdims:
        if axis is None:
            keepdim = (1,) * a.ndim
        else:
            keepdim = list(a.shape)
            for ax in axis:
                keepdim[ax % a.ndim] = 1
            keepdim = tuple(keepdim)

    # Copy a since we need it sorted but without modifying the original array
    if isinstance(axis, int):
        axis = axis,
    if axis is None:
        ap = a.flatten()
        nkeep = 0
    else:
        # Reduce axes from a and put them last
        axis = tuple(ax % a.ndim for ax in axis)
        keep = set(range(a.ndim)) - set(axis)
        nkeep = len(keep)
        for i, s in enumerate(sorted(keep)):
            a = a.swapaxes(i, s)
        ap = a.reshape(a.shape[:nkeep] + (-1,)).copy()

    axis = -1
    ap.sort(axis=axis)
    Nx = ap.shape[axis]
    indices = q * (Nx - 1.)

    if interpolation == 'lower':
        indices = cupy.floor(indices).astype(cupy.int32)
    elif interpolation == 'higher':
        indices = cupy.ceil(indices).astype(cupy.int32)
    elif interpolation == 'midpoint':
        indices = 0.5 * (cupy.floor(indices) + cupy.ceil(indices))
    elif interpolation == 'nearest':
        # TODO(hvy): Implement nearest using around
        raise ValueError('\'nearest\' interpolation is not yet supported. '
                         'Please use any other interpolation method.')
    elif interpolation == 'linear':
        pass
    else:
        raise ValueError('Unexpected interpolation method.\n'
                         'Actual: \'{0}\' not in (\'linear\', \'lower\', '
                         '\'higher\', \'midpoint\')'.format(interpolation))

    if indices.dtype == cupy.int32:
        ret = cupy.rollaxis(ap, axis)
        ret = ret.take(indices, axis=0, out=out)
    else:
        if out is None:
            ret = cupy.empty(ap.shape[:-1] + q.shape, dtype=cupy.float64)
        else:
            ret = cupy.rollaxis(out, 0, out.ndim)

        cupy.ElementwiseKernel(
            'S idx, raw T a, raw int32 offset, raw int32 size', 'U ret',
            '''
            ptrdiff_t idx_below = floor(idx);
            U weight_above = idx - idx_below;

            ptrdiff_t max_idx = size - 1;
            ptrdiff_t offset_bottom = _ind.get()[0] * offset + idx_below;
            ptrdiff_t offset_top = min(offset_bottom + 1, max_idx);

            U diff = a[offset_top] - a[offset_bottom];

            if (weight_above < 0.5) {
                ret = a[offset_bottom] + diff * weight_above;
            } else {
                ret = a[offset_top] - diff * (1 - weight_above);
            }
            ''',
            'percentile_weightnening'
        )(indices, ap, ap.shape[-1] if ap.ndim > 1 else 0, ap.size, ret)
        ret = cupy.rollaxis(ret, -1)  # Roll q dimension back to first axis

    if zerod:
        ret = ret.squeeze(0)
    if keepdims:
        if q.size > 1:
            keepdim = (-1,) + keepdim
        ret = ret.reshape(keepdim)

    return _core._internal_ascontiguousarray(ret)


def _quantile_is_valid(q):
    if cupy.count_nonzero(q < 0.0) or cupy.count_nonzero(q > 1.0):
        return False
    return True


def percentile(a, q, axis=None, out=None, interpolation='linear',
               keepdims=False):
    """Computes the q-th percentile of the data along the specified axis.

    Args:
        a (cupy.ndarray): Array for which to compute percentiles.
        q (float, tuple of floats or cupy.ndarray): Percentiles to compute
            in the range between 0 and 100 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            percentiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        interpolation (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are``lower``, ``higher``, ``midpoint``,
            ``nearest`` and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The percentiles of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.percentile`
    """
    if not isinstance(q, cupy.ndarray):
        q = cupy.asarray(q, dtype='d')
    q = cupy.true_divide(q, 100)
    if not _quantile_is_valid(q):  # synchronize
        raise ValueError('Percentiles must be in the range [0, 100]')
    return _quantile_unchecked(a, q, axis=axis, out=out,
                               interpolation=interpolation, keepdims=keepdims)


def quantile(a, q, axis=None, out=None, interpolation='linear',
             keepdims=False):
    """Computes the q-th quantile of the data along the specified axis.

    Args:
        a (cupy.ndarray): Array for which to compute quantiles.
        q (float, tuple of floats or cupy.ndarray): Quantiles to compute
            in the range between 0 and 1 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            quantiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        interpolation (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are``lower``, ``higher``, ``midpoint``,
            ``nearest`` and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The quantiles of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.quantile`
    """
    if not isinstance(q, cupy.ndarray):
        q = cupy.asarray(q, dtype='d')
    if not _quantile_is_valid(q):  # synchronize
        raise ValueError('Quantiles must be in the range [0, 1]')
    return _quantile_unchecked(a, q, axis=axis, out=out,
                               interpolation=interpolation, keepdims=keepdims)
