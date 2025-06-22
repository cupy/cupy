from __future__ import annotations

import warnings

import numpy

import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.sum`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.sum does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math.sum_auto_dtype
        else:
            func = _math._sum_keep_dtype
        return _fusion_thread_local.call_reduction(
            func, a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return a.sum(axis, dtype, out, keepdims)


def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.prod`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.prod does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math._prod_auto_dtype
        else:
            func = _math._prod_keep_dtype
        return _fusion_thread_local.call_reduction(
            func, a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return a.prod(axis, dtype, out, keepdims)


def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the sum of an array along given axes treating Not a Numbers
    (NaNs) as zero.

    Args:
        a (cupy.ndarray): Array to take sum.
        axis (int or sequence of ints): Axes along which the sum is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nansum`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.nansum does not support `keepdims` in fusion yet.')
        if a.dtype.char in 'FD':
            func = _math._nansum_complex_dtype
        elif dtype is None:
            func = _math._nansum_auto_dtype
        else:
            func = _math._nansum_keep_dtype
        return _fusion_thread_local.call_reduction(
            func, a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return _math._nansum(a, axis, dtype, out, keepdims)


def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    """Returns the product of an array along given axes treating Not a Numbers
    (NaNs) as zero.

    Args:
        a (cupy.ndarray): Array to take product.
        axis (int or sequence of ints): Axes along which the product is taken.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nanprod`

    """
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError(
                'cupy.nanprod does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math._nanprod_auto_dtype
        else:
            func = _math._nanprod_keep_dtype
        return _fusion_thread_local.call_reduction(
            func, a, axis=axis, dtype=dtype, out=out)

    # TODO(okuta): check type
    return _math._nanprod(a, axis, dtype, out, keepdims)


def cumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative sum is taken. If it is not
            specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumsum`

    """
    return _math.scan_core(a, axis, _math.scan_op.SCAN_SUM, dtype, out)


def cumprod(a, axis=None, dtype=None, out=None):
    """Returns the cumulative product of an array along a given axis.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative product is taken. If it is
            not specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.cumprod`

    """
    return _math.scan_core(a, axis, _math.scan_op.SCAN_PROD, dtype, out)


def nancumsum(a, axis=None, dtype=None, out=None):
    """Returns the cumulative sum of an array along a given axis treating Not a
    Numbers (NaNs) as zero.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative sum is taken. If it is not
            specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nancumsum`
    """
    a = _replace_nan(a, 0, out=out)
    return cumsum(a, axis=axis, dtype=dtype, out=out)


def nancumprod(a, axis=None, dtype=None, out=None):
    """Returns the cumulative product of an array along a given axis treating
    Not a Numbers (NaNs) as one.

    Args:
        a (cupy.ndarray): Input array.
        axis (int): Axis along which the cumulative product is taken. If it is
            not specified, the input is flattened.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.nancumprod`
    """
    a = _replace_nan(a, 1, out=out)
    return cumprod(a, axis=axis, dtype=dtype, out=out)


_replace_nan_kernel = cupy._core._kernel.ElementwiseKernel(
    'T a, T val', 'T out', 'if (a == a) {out = a;} else {out = val;}',
    'cupy_replace_nan')


def _replace_nan(a, val, out=None):
    if out is None or a.dtype != out.dtype:
        out = cupy.empty_like(a)
    _replace_nan_kernel(a, val, out)
    return out


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """Calculate the n-th discrete difference along the given axis.

    Args:
        a (cupy.ndarray): Input array.
        n (int): The number of times values are differenced. If zero, the input
            is returned as-is.
        axis (int): The axis along which the difference is taken, default is
            the last axis.
        prepend (int, float, cupy.ndarray): Value to prepend to ``a``.
        append (int, float, cupy.ndarray): Value to append to ``a``.

    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.diff`
    """

    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))

    a = cupy.asanyarray(a)
    nd = a.ndim
    axis = internal._normalize_axis_index(axis, nd)

    combined = []

    if prepend is not None:
        prepend = cupy.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = cupy.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        append = cupy.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = cupy.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = cupy.concatenate(combined, axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = cupy.not_equal if a.dtype == numpy.bool_ else cupy.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])

    return a


def gradient(f, *varargs, axis=None, edge_order=1):
    """Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Args:
        f (cupy.ndarray): An N-dimensional array containing samples of a scalar
            function.
        varargs (list of scalar or array, optional): Spacing between f values.
            Default unitary spacing for all dimensions. Spacing can be
            specified using:

            1. single scalar to specify a sample distance for all dimensions.
            2. N scalars to specify a constant sample distance for each
               dimension. i.e. `dx`, `dy`, `dz`, ...
            3. N arrays to specify the coordinates of the values along each
               dimension of F. The length of the array must match the size of
               the corresponding dimension
            4. Any combination of N scalars/arrays with the meaning of 2. and
               3.

            If `axis` is given, the number of varargs must equal the number of
            axes. Default: 1.
        edge_order ({1, 2}, optional): The gradient is calculated using N-th
            order accurate differences at the boundaries. Default: 1.
        axis (None or int or tuple of ints, optional): The gradient is
            calculated only along the given axis or axes. The default
            (axis = None) is to calculate the gradient for all the axes of the
            input array. axis may be negative, in which case it counts from the
            last to the first axis.

    Returns:
        gradient (cupy.ndarray or list of cupy.ndarray): A set of ndarrays
        (or a single ndarray if there is only one dimension) corresponding
        to the derivatives of f with respect to each dimension. Each
        derivative has the same shape as f.

    .. seealso:: :func:`numpy.gradient`
    """
    f = cupy.asanyarray(f)
    ndim = f.ndim  # number of dimensions
    axes = internal._normalize_axis_indices(axis, ndim, sort_axes=False)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and cupy.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            if cupy.ndim(distances) == 0:
                continue
            elif cupy.ndim(distances) != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )
            if numpy.issubdtype(distances.dtype, numpy.integer):
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(numpy.float64)
            diffx = cupy.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():  # synchronize
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice3 = [slice(None)] * ndim
    slice4 = [slice(None)] * ndim

    otype = f.dtype
    if numpy.issubdtype(otype, numpy.inexact):
        pass
    else:
        # All other types convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        if numpy.issubdtype(otype, numpy.integer):
            f = f.astype(numpy.float64)
        otype = numpy.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )
        # result allocation
        out = cupy.empty_like(f, dtype=otype)

        # spacing for the current axis
        uniform_spacing = cupy.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (
                2.0 * ax_dx
            )
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            dx_sum = dx1 + dx2
            a = -(dx2) / (dx1 * dx_sum)
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * dx_sum)
            # fix the shape for broadcasting
            shape = [1] * ndim
            shape[axis] = -1
            a.shape = b.shape = c.shape = tuple(shape)
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = (a * f[tuple(slice2)] +
                                  b * f[tuple(slice3)] +
                                  c * f[tuple(slice4)])

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                dx_sum = dx1 + dx2
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx_sum))
                b = dx_sum / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx_sum))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = (a * f[tuple(slice2)] +
                                  b * f[tuple(slice3)] +
                                  c * f[tuple(slice4)])

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                dx_sum = dx1 + dx2
                a = (dx2) / (dx1 * (dx_sum))
                b = -dx_sum / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx_sum))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = (a * f[tuple(slice2)] +
                                  b * f[tuple(slice3)] +
                                  c * f[tuple(slice4)])
        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


def ediff1d(arr, to_end=None, to_begin=None):
    """
    Calculates the difference between consecutive elements of an array.

    Args:
        arr (cupy.ndarray): Input array.
        to_end (cupy.ndarray, optional): Numbers to append at the end
            of the returned differences.
        to_begin (cupy.ndarray, optional): Numbers to prepend at the
            beginning of the returned differences.

    Returns:
        cupy.ndarray: New array consisting differences among succeeding
        elements.

    .. seealso:: :func:`numpy.ediff1d`
    """
    if not isinstance(arr, cupy.ndarray):
        raise TypeError('`arr` should be of type cupy.ndarray')

    # to flattened array.
    arr = arr.ravel()

    # to ensure the dtype of the output array is same as that of input.
    dtype_req = arr.dtype

    # if none optional cases are given
    if to_begin is None and to_end is None:
        return arr[1:] - arr[:-1]

    if to_begin is None:
        l_begin = 0
    else:
        if not isinstance(to_begin, cupy.ndarray):
            raise TypeError('`to_begin` should be of type cupy.ndarray')
        if not cupy.can_cast(to_begin, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_begin` must be compatible "
                            "with input `arr` under the `same_kind` rule.")

        to_begin = to_begin.ravel()
        l_begin = len(to_begin)

    if to_end is None:
        l_end = 0
    else:
        if not isinstance(to_end, cupy.ndarray):
            raise TypeError('`to_end` should be of type cupy.ndarray')
        if not cupy.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `arr` under the `same_kind` rule.")

        to_end = to_end.ravel()
        l_end = len(to_end)

    # calculating using in place operation
    l_diff = max(len(arr) - 1, 0)
    result = cupy.empty(l_diff + l_begin + l_end, dtype=arr.dtype)
    # Cupy does not support subclassing a ndarray
    # result = arr.__array_wrap__(result)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    cupy.subtract(arr[1:], arr[:-1], result[l_begin:l_begin + l_diff])
    return result


# TODO(okuta): Implement cross


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along the given axis.

    Args:
        y (cupy.ndarray): Input array to integrate.
        x (cupy.ndarray): Sample points over which to integrate. If None equal
            spacing `dx` is assumed.
        dx (float): Spacing between sample points, used if `x` is None, default
            is 1.
        axis (int): The axis along which the integral is taken, default is
            the last axis.

    Returns:
        cupy.ndarray: Definite integral as approximated by the trapezoidal
        rule.

    .. seealso:: :func:`numpy.trapezoid`
    """
    if not isinstance(y, cupy.ndarray):
        raise TypeError('`y` should be of type cupy.ndarray')

    if x is None:
        d = dx
    else:
        if not isinstance(x, cupy.ndarray):
            raise TypeError('`x` should be of type cupy.ndarray')
        if x.ndim == 1:
            d = diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = cupy.add.reduce(product, axis)
    return ret


def product(a, axis=None, dtype=None, out=None, keepdims=False):
    warnings.warn('Please use `prod` instead.', DeprecationWarning)
    return prod(a, axis, dtype, out, keepdims)


def cumproduct(a, axis=None, dtype=None, out=None):
    warnings.warn('Please use `cumprod` instead.', DeprecationWarning)
    return cumprod(a, axis, dtype, out)
