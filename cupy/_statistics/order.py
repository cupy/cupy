from __future__ import annotations

import warnings

import numpy

import cupy
from cupy import _core
from cupy import _util
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content

# Quantile method parameters (alpha, beta) from Hyndman & Fan (1996)
# Used for continuous interpolation methods in percentile/quantile
_QUANTILE_PARAMS = {
    "hazen": (0.5, 0.5),  # H&F type 5
    "weibull": (0, 0),  # H&F type 6
    "median_unbiased": (1 / 3, 1 / 3),  # H&F type 8
    "normal_unbiased": (3 / 8, 3 / 8),  # H&F type 9
}


@_util.memoize()
def _get_percentile_weightnening_kernel():
    return cupy.ElementwiseKernel(
        "S idx, raw T a, int64 offset_, int64 size_",
        "U ret",
        """
        using index_t = decltype(a)::index_t;
        index_t offset = static_cast<index_t>(offset_);
        index_t size = static_cast<index_t>(size_);

        index_t idx_below = floor(idx);
        U weight_above = idx - idx_below;

        index_t max_idx = size - 1;
        index_t offset_bottom = _ind.get()[0] * offset + idx_below;
        index_t offset_top = min(offset_bottom + 1, max_idx);

        U diff = a[offset_top] - a[offset_bottom];

        if (weight_above < 0.5) {
            ret = a[offset_bottom] + diff * weight_above;
        } else {
            ret = a[offset_top] - diff * (1 - weight_above);
        }
        """,
        "cupy_percentile_weightnening",
    )


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
                "cupy.amin does not support `keepdims` in fusion yet."
            )
        return _fusion_thread_local.call_reduction(
            _statistics.amin, a, axis=axis, out=out
        )

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
                "cupy.amax does not support `keepdims` in fusion yet."
            )
        return _fusion_thread_local.call_reduction(
            _statistics.amax, a, axis=axis, out=out
        )

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
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
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
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
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


def _empty_nan_reduction(a, axis, out, keepdims, dtype):
    # All-NaN result of reducing an empty ``a``, shaped like a mean reduction.
    if axis is None:
        shape = (1,) * a.ndim if keepdims else ()
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(ax % a.ndim for ax in axis)
        if keepdims:
            shape = tuple(
                1 if ax in axis else size for ax, size in enumerate(a.shape)
            )
        else:
            shape = tuple(
                size for ax, size in enumerate(a.shape) if ax not in axis
            )
    if out is None:
        return cupy.full(shape, cupy.nan, dtype=dtype)
    out[...] = cupy.nan
    return out


def _quantile_unchecked(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    nan_sensitive=False,
):
    dtype = cupy.result_type(a, q)
    if nan_sensitive and a.size == 0:
        # NumPy short-circuits empty input to ``nanmean`` (see
        # ``numpy.lib._nanfunctions_impl._nanquantile_unchecked``), so ``q`` is
        # dropped from the result shape. Interpolation has nothing to work
        # with here, and the reshape below cannot infer the reduced dimension
        # of an empty array. ``cupy.nanmean`` itself is not reusable: it
        # raises on a zero-size reduction axis where NumPy returns NaN. The
        # dtype is ``a.dtype`` because ``nanmean`` keeps the inexact dtype of
        # its input.
        return _empty_nan_reduction(a, axis, out, keepdims, a.dtype)

    q = cupy.asarray(q)

    if q.ndim == 0:
        q = q[None]
        zerod = True
    else:
        zerod = False
    if q.ndim > 1:
        raise ValueError(
            "Expected q to have a dimension of 1.\nActual: {} != 1".format(
                q.ndim
            )
        )
    if isinstance(axis, int):
        axis = (axis,)
    if keepdims:
        if axis is None:
            keepdim = (1,) * a.ndim
        else:
            keepdim = list(a.shape)
            for ax in axis:
                keepdim[ax % a.ndim] = 1
            keepdim = tuple(keepdim)
    if axis is None:
        if overwrite_input:
            ap = a.ravel()
        else:
            ap = a.flatten()
        nkeep = 0
    else:
        # Reduce axes from a and put them last
        axis = tuple(ax % a.ndim for ax in axis)
        keep = set(range(a.ndim)) - set(axis)
        nkeep = len(keep)
        for i, s in enumerate(sorted(keep)):
            a = a.swapaxes(i, s)
        if overwrite_input:
            ap = a.reshape(a.shape[:nkeep] + (-1,))
        else:
            ap = a.reshape(a.shape[:nkeep] + (-1,)).copy()

    axis = -1
    ap.sort(axis=axis)
    Nx = ap.shape[axis]
    if nan_sensitive:
        # NaNs are sorted to the end of each slice, so the number of valid
        # (non-NaN) elements of a slice is just the count of its non-NaN
        # values. Slices without any valid element are clamped to a single
        # element so that their result is the (NaN) value at index 0.
        # ``n_obs`` keeps the per-slice axis (no ``keepdims``) so that
        # ``indices`` ends up shaped ``q.shape + nkeep_shape``, the same
        # layout the caller expects back from ``numpy.nanquantile``.
        n_obs = (~cupy.isnan(ap)).sum(axis=axis)
        # Match the dtype arithmetic of the non-NaN branch below, where the
        # slice length is a Python int, so that a float32 ``q`` stays
        # float32 here as well. Integer ``q`` cannot be used as a stand-in: a
        # slice with more valid elements than that dtype can hold (e.g. 300
        # in int8) would wrap around to a negative count.
        n_obs = n_obs.astype(
            q.dtype if q.dtype.kind == "f" else cupy.float64, copy=False
        )
        # ``max_index`` is int64: a reduced slice can hold more than 2**31
        # valid elements, and a wrapped index would silently clip against a
        # negative bound and gather the wrong slice. Where int32 indices are
        # genuinely required, the cast happens locally.
        max_index = cupy.maximum(n_obs - 1, 0).astype(cupy.int64)
        # Give ``q`` and ``n_obs`` a trailing broadcast axis each, so that
        # every index formula below yields ``q.shape + nkeep_shape`` exactly
        # as in the non-NaN branch, where ``q`` broadcasts against a scalar.
        q_ndim, n_obs_ndim = q.ndim, n_obs.ndim
        q = q.reshape(q.shape + (1,) * n_obs_ndim)
        n_obs = n_obs.reshape((1,) * q_ndim + n_obs.shape)
    else:
        n_obs = Nx
        max_index = Nx - 1
    indices = q * (n_obs - 1.0)

    if method in [
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
    ]:
        # TODO(takagi) Implement new methods introduced in NumPy 1.22
        raise ValueError(
            f"'{method}' method is not yet supported. "
            "Please use any other method."
        )
    elif method in _QUANTILE_PARAMS:
        alpha, beta = _QUANTILE_PARAMS[method]
        indices = q * (n_obs - alpha - beta + 1) + alpha - 1
        indices = cupy.clip(indices, 0, max_index)
    elif method == "lower":
        indices = cupy.floor(indices).astype(cupy.int32)
    elif method == "higher":
        indices = cupy.ceil(indices).astype(cupy.int32)
    elif method == "midpoint":
        indices = 0.5 * (cupy.floor(indices) + cupy.ceil(indices))
    elif method == "nearest":
        indices = cupy.around(indices).astype(cupy.int32)
    elif method == "inverted_cdf":
        indices = cupy.clip(
            cupy.ceil(q * n_obs).astype(cupy.int32) - 1, 0, max_index
        )
    elif method == "linear":
        pass
    else:
        raise ValueError(
            "Unexpected interpolation method.\n"
            "Actual: '{}' not in ('linear', 'lower',"
            " 'higher','midpoint', 'inverted_cdf', "
            "'nearest')".format(method)
        )

    if nan_sensitive:
        # Keeps the indices inside each slice's valid range (all-NaN slices
        # are clamped to 0). The bound is cast to the index dtype so that
        # the clip does not promote float indices.
        max_index_f = max_index.astype(indices.dtype, copy=False)
        indices = cupy.clip(indices, 0, max_index_f)

        # ``indices`` holds one value per (q, slice) pair here, so neither
        # ``take`` along axis 0 nor the flat weightening kernel - both of
        # which assume a single index per q - can be used. Gather through the
        # flat buffer instead, in the same ``nkeep_shape + (nq,)`` layout the
        # kernel writes, and interpolate with the kernel's very formula (this
        # is the one place where the formula is duplicated instead of reused:
        # the kernel takes one index per q, this path needs one per
        # (q, slice)).
        idx = cupy.moveaxis(indices, 0, -1)
        # ``base`` holds absolute offsets into the flat buffer, so it must be
        # int64: arrays with more than 2**31 elements (e.g. 8.6 GB of float32)
        # would otherwise wrap around and gather the wrong slice.
        off = (
            cupy.arange(ap.size // Nx, dtype=cupy.int64).reshape(
                ap.shape[:-1] + (1,)
            )
            * Nx
        )
        base = off + idx.astype(cupy.int64)
        flat = ap.ravel()
        if indices.dtype.kind in "iu":
            res = flat.take(base)
        else:
            mx = max_index.reshape(ap.shape[:-1] + (1,))
            below = cupy.floor(idx)
            # Mirror the kernel exactly: the weight is computed in the output
            # dtype, the difference in the input dtype T.
            weight = (idx - below).astype(dtype, copy=False)
            low = flat.take(base)
            high = flat.take(cupy.minimum(base + 1, off + mx))
            diff = (high - low).astype(dtype, copy=False)
            low = low.astype(dtype, copy=False)
            high = high.astype(dtype, copy=False)
            res = cupy.where(
                weight < 0.5, low + diff * weight, high - diff * (1 - weight)
            )
        if out is None:
            ret = cupy.rollaxis(res, -1)  # Roll q dimension to first axis
        else:
            ret = cupy.rollaxis(out, 0, out.ndim)
            ret[...] = res
            ret = cupy.rollaxis(ret, -1)  # Roll q dimension back to first axis
    elif indices.dtype == cupy.int32:
        ret = cupy.rollaxis(ap, axis)
        ret = ret.take(indices, axis=0, out=out)
    else:
        if out is None:
            ret = cupy.empty(ap.shape[:-1] + q.shape, dtype=dtype)
        else:
            ret = cupy.rollaxis(out, 0, out.ndim)

        _get_percentile_weightnening_kernel()(
            indices, ap, ap.shape[-1] if ap.ndim > 1 else 0, ap.size, ret
        )
        ret = cupy.rollaxis(ret, -1)  # Roll q dimension back to first axis

    if zerod:
        ret = ret.squeeze(0)
    if keepdims:
        if q.size > 1:
            keepdim = (-1,) + keepdim
        ret = ret.reshape(keepdim)

    return _core._internal_ascontiguousarray(ret)


def _quantile_is_valid(q):
    xp = cupy if isinstance(q, cupy.ndarray) else numpy
    return xp.count_nonzero(0.0 <= q) and xp.count_nonzero(q <= 1.0)


def _has_all_nan_slice(a, axis):
    # Returns whether ``a`` has a slice along ``axis`` that is all NaN.
    # The check is done on the input rather than on the result, because the
    # result may be a user-supplied ``out`` buffer that already holds NaNs.
    if a.size == 0:
        return False
    if a.ndim == 0:
        return bool(content.isnan(a))  # synchronize!
    if axis is None:
        axis = tuple(range(a.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    axis = tuple(ax % a.ndim for ax in axis)
    n_obs = cupy.sum(~content.isnan(a), axis=axis)
    return bool(cupy.any(n_obs == 0))  # synchronize!


def percentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    """Computes the q-th percentile of the data along the specified axis.

    Args:
        a (cupy.ndarray): Array for which to compute percentiles.
        q (float, tuple of floats or cupy.ndarray): Percentiles to compute
            in the range between 0 and 100 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            percentiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If True, then allow the input array `a`
            to be modified by the intermediate calculations, to save
            memory. In this case, the contents of the input `a` after this
            function completes is undefined.
        method (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are ``lower``, ``higher``, ``midpoint``,
            ``nearest``, ``inverted_cdf``  and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        interpolation (str): Deprecated name for the method keyword argument.

    Returns:
        cupy.ndarray: The percentiles of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.percentile`
    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "percentile"
        )
    if isinstance(q, (tuple, list)):
        # float is intentionally excluded here to compute the correct output
        # dtype in _quantile_unchecked
        q = numpy.asarray(q)
    q = q / 100
    if not _quantile_is_valid(q):  # synchronize if `q` is of cupy.ndarray
        raise ValueError("Percentiles must be in the range [0, 100]")
    return _quantile_unchecked(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )


def nanpercentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
):
    """Computes the q-th percentile of the data along the specified axis,
    while ignoring NaNs.

    When there is a slice whose elements are all NaN, a :class:`RuntimeWarning`
    is raised and NaN is returned.

    Args:
        a (cupy.ndarray): Array for which to compute percentiles.
        q (float, tuple of floats or cupy.ndarray): Percentiles to compute
            in the range between 0 and 100 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            percentiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If True, then allow the input array `a`
            to be modified by the intermediate calculations, to save
            memory. In this case, the contents of the input `a` after this
            function completes is undefined.
        method (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are ``lower``, ``higher``, ``midpoint``,
            ``nearest``, ``inverted_cdf``  and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The percentiles of ``a``, along the axis if specified.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.nanpercentile`
    """
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    if a.dtype.char not in "efdFD":
        # NaNs cannot occur, so this is exactly `percentile`.
        return percentile(
            a,
            q,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=method,
            keepdims=keepdims,
        )
    if isinstance(q, (tuple, list)):
        # float is intentionally excluded here to compute the correct output
        # dtype in _quantile_unchecked
        q = numpy.asarray(q)
    q = q / 100
    if not _quantile_is_valid(q):  # synchronize if `q` is of cupy.ndarray
        raise ValueError("Percentiles must be in the range [0, 100]")
    # TODO(niboshi): Avoid synchronization.
    res = _quantile_unchecked(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        nan_sensitive=True,
    )
    if _has_all_nan_slice(a, axis):  # synchronize!
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
    return res


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    """Computes the q-th quantile of the data along the specified axis.

    Args:
        a (cupy.ndarray): Array for which to compute quantiles.
        q (float, tuple of floats or cupy.ndarray): Quantiles to compute
            in the range between 0 and 1 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            quantiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If True, then allow the input array `a`
            to be modified by the intermediate calculations, to save
            memory. In this case, the contents of the input `a` after this
            function completes is undefined.
        method (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are ``lower``, ``higher``, ``midpoint``,
            ``nearest``, ``inverted_cdf`` and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        interpolation (str): Deprecated name for the method keyword argument.

    Returns:
        cupy.ndarray: The quantiles of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.quantile`
    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "quantile"
        )
    if isinstance(q, (tuple, list)):
        # float is intentionally excluded here to compute the correct output
        # dtype in _quantile_unchecked
        q = numpy.asarray(q)
    if not _quantile_is_valid(q):  # synchronize if `q` is of cupy.ndarray
        raise ValueError("Quantiles must be in the range [0, 1]")
    return _quantile_unchecked(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )


def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
):
    """Computes the q-th quantile of the data along the specified axis,
    while ignoring NaNs.

    When there is a slice whose elements are all NaN, a :class:`RuntimeWarning`
    is raised and NaN is returned.

    Args:
        a (cupy.ndarray): Array for which to compute quantiles.
        q (float, tuple of floats or cupy.ndarray): Quantiles to compute
            in the range between 0 and 1 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            quantiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If True, then allow the input array `a`
            to be modified by the intermediate calculations, to save
            memory. In this case, the contents of the input `a` after this
            function completes is undefined.
        method (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are ``lower``, ``higher``, ``midpoint``,
            ``nearest``, ``inverted_cdf`` and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The quantiles of ``a``, along the axis if specified.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.nanquantile`
    """
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")
    if a.dtype.char not in "efdFD":
        # NaNs cannot occur, so this is exactly `quantile`.
        return quantile(
            a,
            q,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=method,
            keepdims=keepdims,
        )
    if isinstance(q, (tuple, list)):
        # float is intentionally excluded here to compute the correct output
        # dtype in _quantile_unchecked
        q = numpy.asarray(q)
    if not _quantile_is_valid(q):  # synchronize if `q` is of cupy.ndarray
        raise ValueError("Quantiles must be in the range [0, 1]")
    # TODO(niboshi): Avoid synchronization.
    res = _quantile_unchecked(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        nan_sensitive=True,
    )
    if _has_all_nan_slice(a, axis):  # synchronize!
        warnings.warn("All-NaN slice encountered", RuntimeWarning)
    return res


# Borrowd from NumPy
def _check_interpolation_as_method(method, interpolation, fname):
    # Deprecated NumPy 1.22, 2021-11-08
    warnings.warn(
        f"the `interpolation=` argument to {fname} was renamed to "
        "`method=`, which has additional options.\n"
        "Users of the modes 'nearest', 'lower', 'higher', or "
        "'midpoint' are encouraged to review the method they. "
        "(Deprecated NumPy 1.22)",
        DeprecationWarning,
        stacklevel=3,
    )
    if method != "linear":
        # sanity check, we assume this basically never happens
        raise TypeError(
            "You shall not pass both `method` and `interpolation`!\n"
            "(`interpolation` is Deprecated in favor of `method`)"
        )
    return interpolation
