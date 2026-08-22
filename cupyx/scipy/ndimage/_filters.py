from __future__ import annotations

import numpy

import cupy

from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic


def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0,
              *, axes=None):
    """Multi-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None, optional):  If None, `input` is filtered
            along all axes. Otherwise, `input` is filtered along the specified
            axes. When `axes` is specified, any tuples used for `mode` or
            `origin` must match the length of `axes`. The ith entry in any of
            these tuples corresponds to the ith entry in `axes`.

    Returns:
        cupy.ndarray: The result of correlate.

    .. seealso:: :func:`scipy.ndimage.correlate`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve(
        input, weights, output, mode, cval, origin, False, axes
    )


def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0,
             *, axes=None):
    """Multi-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None, optional):  If None, `input` is filtered
            along all axes. Otherwise, `input` is filtered along the specified
            axes. When `axes` is specified, any tuples used for `mode` or
            `origin` must match the length of `axes`. The ith entry in any of
            these tuples corresponds to the ith entry in `axes`.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.convolve`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve(
        input, weights, output, mode, cval, origin, True, axes
    )


def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0):
    """One-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the 1D correlation.

    .. seealso:: :func:`scipy.ndimage.correlate1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights, origins = _filters_core._convert_1d_args(input.ndim, weights,
                                                      origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins)


def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
               origin=0):
    """One-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the 1D convolution.

    .. seealso:: :func:`scipy.ndimage.convolve1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights, origins = _filters_core._convert_1d_args(input.ndim, weights,
                                                      origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins,
                                  True)


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution=False, axes=None):
    axes, weights, origins, modes, int_type = _filters_core._check_nd_args(
        input, weights, mode, origin, axes=axes)
    if weights.size == 0:
        return cupy.zeros_like(input)

    for mode in modes:
        _util._check_cval(mode, cval, _util._is_integer_output(output, input))

    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    elif weights.dtype.kind == "c":
        # numpy.correlate conjugates weights rather than input.
        weights = weights.conj()
    weights_dtype = _util._get_weights_dtype(input, weights)
    offsets = _filters_core._origins_to_offsets(origins, weights.shape)
    kernel = _get_correlate_kernel(modes, weights.shape, int_type,
                                   offsets, cval)
    output = _filters_core._call_kernel(kernel, input, weights, output,
                                        weights_dtype=weights_dtype)
    return output


@cupy._util.memoize(for_each_device=True)
def _get_correlate_kernel(modes, w_shape, int_type, offsets, cval):
    return _filters_core._generate_nd_kernel(
        'correlate',
        'W sum = (W)0;',
        'sum += cast<W>({value}) * wval;',
        'y = cast<Y>(sum);',
        modes, w_shape, int_type, offsets, cval, ctype='W')


def _run_1d_correlates(input, axes, params, get_weights, output, modes, cval,
                       origin=0):
    """
    Enhanced version of _run_1d_filters that uses correlate1d as the filter
    function. The params are a list of values to pass to the get_weights
    callable given. If duplicate param values are found, the weights are
    reused from the first invocation of get_weights. The get_weights callable
    must return a 1D array of weights to give to correlate1d.
    """
    wghts = {}
    for param in params:
        if param not in wghts:
            wghts[param] = get_weights(param)
    wghts = [wghts[param] for param in params]
    return _filters_core._run_1d_filters(
        [None if w is None else correlate1d for w in wghts],
        input, axes, wghts, output, modes, cval, origin)


def uniform_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """One-dimensional uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a uniform
    filter of the given size.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the uniform filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.uniform_filter1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.full(size, 1 / size, dtype=weights_dtype)
    return correlate1d(input, weights, axis, output, mode, cval, origin)


def uniform_filter(input, size=3, output=None, mode="reflect", cval=0.0,
                   origin=0, axes=None):
    """Multi-dimensional uniform filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): Lengths of the uniform filter for each
            dimension. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.uniform_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of ``0`` is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size``,
            ``mode`` and/or ``origin`` must match the length of ``axes``. The
            ith entry in any of these tuples corresponds to the ith entry in
            ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.uniform_filter`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sizes = _util._fix_sequence_arg(size, num_axes, 'size', int)
    origins = _util._fix_sequence_arg(origin, num_axes, 'origin', int)
    modes = _util._fix_sequence_arg(mode, num_axes, 'mode', str)

    weights_dtype = _util._init_weights_dtype(input)

    def get(size, dtype=weights_dtype):
        return None if size <= 1 else cupy.full(size, 1 / size, dtype=dtype)

    return _run_1d_correlates(input, axes, sizes, get, output, modes, cval,
                              origins)


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter along the given axis.

    The lines of the array along the given axis are filtered with a Gaussian
    filter of the given standard deviation.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar): Standard deviation for Gaussian kernel.
        axis (int): The axis of input along which to calculate. Default is -1.
        order (int): An order of ``0``, the default, corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_filter1d`

    .. note::
        The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
        `radius` is None, a default ``radius = round(truncate * sigma)`` will
        be used.

        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    radius = int(float(truncate) * float(sigma) + 0.5)
    weights_dtype = _util._init_weights_dtype(input)
    weights = _gaussian_kernel1d(
        sigma, int(order), radius, dtype=weights_dtype
    )
    return correlate1d(input, weights, axis, output, mode, cval)


def gaussian_filter(input, sigma, order=0, output=None, mode="reflect",
                    cval=0.0, truncate=4.0, radius=None, axes=None):
    """Multi-dimensional Gaussian filter.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        order (int or sequence of scalar): An order of ``0``, the default,
            corresponds to convolution with a Gaussian kernel. A positive order
            corresponds to convolution with that derivative of a Gaussian. A
            single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.gaussian_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.
        radius (int, sequence of int, or None): Radius of the Gaussian kernel.
            The radius are given for each axis as a sequence, or as a single
            number, in which case it is equal for all axes. If specified, the
            size of the kernel along each axis will be ``2*radius + 1``, and
            `truncate` is ignored. Default is ``None``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``sigma``,
            ``order``, ``mode`` and/or ``radius`` must match the length of
            ``axes``. The ith entry in any of these tuples corresponds to the
            ith entry in ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_filter`

    .. note::
        The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
        `radius` is None, a default ``radius = round(truncate * sigma)`` will
        be used.

        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sigmas = _util._fix_sequence_arg(sigma, num_axes, 'sigma', float)
    sigma_threshold = 1e-15
    if num_axes == 0 or all(s < sigma_threshold for s in sigmas):
        if output is None:
            return input.copy()
        else:
            output = _util._get_output(output, input)
            output[:] = input
            return output
    orders = _util._fix_sequence_arg(order, num_axes, 'order', int)
    modes = _util._fix_sequence_arg(mode, num_axes, 'mode', str)
    radiuses = _util._fix_sequence_arg(radius, num_axes, 'radius')
    truncate = float(truncate)
    weights_dtype = _util._init_weights_dtype(input)

    # omit any axes with sigma ~= 0.0
    params = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
              for ii in range(num_axes) if sigmas[ii] > 1e-15]
    # update arguments in case any were filtered out due to sigma ~= 0.0
    axes, sigmas, orders, modes, radiuses = zip(*params)

    def get(param):
        _, sigma, order, _, radius = param
        if radius is None:
            radius = int(truncate * float(sigma) + 0.5)
        if radius <= 0:
            return None
        return _gaussian_kernel1d(sigma, order, radius, dtype=weights_dtype)

    return _run_1d_correlates(input, axes, params, get, output, modes, cval, 0)


def _gaussian_kernel1d(sigma, order, radius, dtype=cupy.float64):
    """
    Computes a 1-D Gaussian correlation kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')

    if cupy.cuda.get_current_stream().is_capturing():
        # CUDA graph capture can't have H2D operations.
        # Perform all ops on the device.
        xp = cupy
    else:
        xp = numpy

    sigma2 = sigma * sigma
    x = xp.arange(-radius, radius+1)
    phi_x = xp.exp(-0.5 / sigma2 * x ** 2)
    phi_x /= phi_x.sum()

    if order == 0:
        return cupy.asarray(phi_x)

    # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
    # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
    # p'(x) = -1 / sigma ** 2
    # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
    # coefficients of q(x)
    exponent_range = xp.arange(order + 1)
    q = xp.zeros(order + 1)
    q[0] = 1
    D = xp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
    P = xp.diag(xp.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1], dtype=dtype)


def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Compute a Prewitt filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.prewitt`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.ones(3, dtype=weights_dtype)
    return _prewitt_or_sobel(input, axis, output, mode, cval, weights)


def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Compute a Sobel filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.sobel`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = _util._init_weights_dtype(input)
    return _prewitt_or_sobel(input, axis, output, mode, cval,
                             cupy.array([1, 2, 1], dtype=weights_dtype))


def _prewitt_or_sobel(input, axis, output, mode, cval, weights):
    axis = internal._normalize_axis_index(axis, input.ndim)

    def get(is_diff):
        return cupy.array([-1, 0, 1], dtype=weights.dtype) if is_diff else weights  # noqa

    axes = tuple(range(input.ndim))
    modes = (mode,) * input.ndim
    return _run_1d_correlates(input, axes,
                              [a == axis for a in range(input.ndim)], get,
                              output, modes, cval)


def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0, extra_arguments=(), extra_keywords=None, *,
                    axes=None):
    """Multi-dimensional Laplace filter using a provided second derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative2 (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative2(input, axis, output, mode, cval,
                            *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.generic_laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    if extra_keywords is None:
        extra_keywords = {}
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    output = _util._get_output(output, input)
    if num_axes > 0:
        modes = _util._fix_sequence_arg(mode, num_axes, 'mode',
                                        _util._check_mode)
        derivative2(input, axes[0], output, modes[0], cval,
                    *extra_arguments, **extra_keywords)
        if num_axes > 1:
            tmp = _util._get_output(output.dtype, input)
            for i in range(1, num_axes):
                derivative2(input, axes[i], tmp, modes[i], cval,
                            *extra_arguments, **extra_keywords)
                output += tmp
    else:
        _core.elementwise_copy(input, output)
    return output


def laplace(input, output=None, mode="reflect", cval=0.0, *, axes=None):
    """Multi-dimensional Laplace filter based on approximate second
    derivatives.

    Args:
        input (cupy.ndarray): The input array.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.array([1, -2, 1], dtype=weights_dtype)

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, weights, axis, output, mode, cval)

    return generic_laplace(input, derivative2, output, mode, cval, axes=axes)


def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, *, axes=None, **kwargs):
    """Multi-dimensional Laplace filter using Gaussian second derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `sigma` or `mode` tuples are provided, their length must match
            the number of axes.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)

    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sigma = _util._fix_sequence_arg(sigma, num_axes, "sigma", float)
    if num_axes < input.ndim:
        # set sigma = 0 for any axes not being filtered
        sigma_temp = [0,] * input.ndim
        for s, ax in zip(sigma, axes):
            sigma_temp[ax] = s
        sigma = sigma_temp

    return generic_laplace(input, derivative2, output, mode, cval,
                           extra_arguments=(sigma,), extra_keywords=kwargs,
                           axes=axes)


def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None, *,
                               axes=None):
    """Multi-dimensional gradient magnitude filter using a provided derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative(input, axis, output, mode, cval,
                           *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.generic_gradient_magnitude`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    axes = _util._check_axes(axes, ndim)
    num_axes = len(axes)
    modes = _util._fix_sequence_arg(mode, num_axes, 'mode',
                                    _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        _core.elementwise_copy(input, output)
        return output
    derivative(input, axes[0], output, modes[0], cval,
               *extra_arguments, **extra_keywords)
    output *= output
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, num_axes):
            derivative(input, axes[i], tmp, modes[i], cval,
                       *extra_arguments, **extra_keywords)
            tmp *= tmp
            output += tmp
    return cupy.sqrt(output, output, casting='unsafe')


def gaussian_gradient_magnitude(input, sigma, output=None, mode="reflect",
                                cval=0.0, *, axes=None, **kwargs):
    """Multi-dimensional gradient magnitude using Gaussian derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_gradient_magnitude`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    def derivative(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)
    return generic_gradient_magnitude(input, derivative, output, mode, cval,
                                      axes=axes)


def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0, axes=None):
    """Multi-dimensional minimum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.minimum_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size``,
            ``mode`` and/or ``origin`` must match the length of ``axes``. The
            ith entry in any of these tuples corresponds to the ith entry in
            ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 'min', axes)


def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0, axes=None):
    """Multi-dimensional maximum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.minimum_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size``,
            ``mode`` and/or ``origin`` must match the length of ``axes``. The
            ith entry in any of these tuples corresponds to the ith entry in
            ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 'max', axes)


def _min_or_max_filter(input, size, ftprnt, structure, output, mode, cval,
                       origin, func, axes):
    # structure is used by morphology.grey_erosion() and grey_dilation()
    # and not by the regular min/max filters

    if isinstance(ftprnt, tuple) and size is None:
        size = ftprnt
        ftprnt = None

    axes, ftprnt, origins, modes, int_type = _filters_core._check_nd_args(
        input, ftprnt, mode, origin, 'footprint', sizes=size, axes=axes,
        raise_on_zero_size_weight=True)
    num_axes = len(axes)
    sizes, ftprnt, structure = _filters_core._check_size_footprint_structure(
        num_axes, size, ftprnt, structure)
    if cval is cupy.nan:
        raise NotImplementedError("NaN cval is unsupported")

    if sizes is not None:
        # Separable filter, run as a series of 1D filters
        fltr = minimum_filter1d if func == 'min' else maximum_filter1d
        return _filters_core._run_1d_filters(
            [fltr if size > 1 else None for size in sizes],
            input, axes, sizes, output, modes, cval, origins)

    if ftprnt.size == 0:
        return cupy.zeros_like(input)

    if num_axes < input.ndim:
        # expand origins ,footprint and structure if num_axes < input.ndim
        ftprnt = _util._expand_footprint(input.ndim, axes, ftprnt)
        origins = _util._expand_origin(input.ndim, axes, origin)
        modes = tuple(_util._expand_mode(input.ndim, axes, modes))

    if structure is not None:
        structure = _util._expand_footprint(
            input.ndim, axes, structure, footprint_name="structure"
        )

    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(modes, ftprnt.shape, func,
                                    offsets, float(cval), int_type,
                                    has_structure=structure is not None,
                                    has_central_value=bool(ftprnt[offsets]))
    return _filters_core._call_kernel(kernel, input, ftprnt, output,
                                      structure, weights_dtype=bool)


def minimum_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """Compute the minimum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the minimum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, 'min')


def maximum_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """Compute the maximum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the maximum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, 'max')


def _min_or_max_1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0,
                   origin=0, func='min'):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origin = _filters_core._convert_1d_args(input.ndim, ftprnt,
                                                    origin, axis)
    axes, ftprnt, origins, modes, int_type = _filters_core._check_nd_args(
        input, ftprnt, mode, origin, 'footprint', axes=None)
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(modes, ftprnt.shape, func, offsets,
                                    float(cval), int_type, has_weights=False)
    return _filters_core._call_kernel(kernel, input, None, output,
                                      weights_dtype=bool)


@cupy._util.memoize(for_each_device=True)
def _get_min_or_max_kernel(modes, w_shape, func, offsets, cval, int_type,
                           has_weights=True, has_structure=False,
                           has_central_value=True):
    # When there are no 'weights' (the footprint, for the 1D variants) then
    # we need to make sure intermediate results are stored as doubles for
    # consistent results with scipy.
    ctype = 'X' if has_weights else 'double'
    value = '{value}'
    if not has_weights:
        value = 'cast<double>({})'.format(value)

    # Having a non-flat structure biases the values
    if has_structure:
        value += ('-' if func == 'min' else '+') + 'cast<X>(sval)'

    if has_central_value:
        pre = '{} value = x[i];'
        found = 'value = {func}({value}, value);'
    else:
        # If the central pixel is not included in the footprint we cannot
        # assume `x[i]` is not below the min or above the max and thus cannot
        # seed with that value. Instead we keep track of having set `value`.
        pre = '{} value; bool set = false;'
        found = 'value = set ? {func}({value}, value) : {value}; set=true;'

    return _filters_core._generate_nd_kernel(
        func, pre.format(ctype),
        found.format(func=func, value=value), 'y = cast<Y>(value);',
        modes, w_shape, int_type, offsets, cval, ctype=ctype,
        has_weights=has_weights, has_structure=has_structure)


def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0, axes=None):
    """Multi-dimensional rank filter.

    Args:
        input (cupy.ndarray): The input array.
        rank (int): The rank of the element to get. Can be negative to count
            from the largest value, e.g. ``-1`` indicates the largest value.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = int(rank)
    return _rank_filter(input, lambda fs: rank+fs if rank < 0 else rank,
                        size, footprint, output, mode, cval, origin, axes)


def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0, axes=None):
    """Multi-dimensional median filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(input, lambda fs: fs//2,
                        size, footprint, output, mode, cval, origin, axes)


def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0,
                      axes=None):
    """Multi-dimensional percentile filter.

    Args:
        input (cupy.ndarray): The input array.
        percentile (scalar): The percentile of the element to get (from ``0``
            to ``100``). Can be negative, thus ``-20`` equals ``80``.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.percentile_filter`
    """
    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError('invalid percentile')
    if percentile == 100.0:
        def get_rank(fs):
            return fs - 1
    else:
        def get_rank(fs):
            return int(float(fs) * percentile / 100.0)
    return _rank_filter(input, get_rank,
                        size, footprint, output, mode, cval, origin, axes)


def _rank_filter(input, get_rank, size=None, footprint=None, output=None,
                 mode="reflect", cval=0.0, origin=0, axes=None):
    if cval is cupy.nan:
        raise NotImplementedError("NaN cval is unsupported")
    axes = _util._check_axes(axes, input.ndim)
    # generate explicit footprint matching axes size
    default_footprint = footprint is None
    _, footprint, _ = _filters_core._check_size_footprint_structure(
        len(axes), size, footprint, None, force_footprint=True)
    if footprint.size == 0:
        return cupy.zeros_like(input)
    # check remaining arguments and update based on axes
    axes, footprint, origins, modes, int_type = _filters_core._check_nd_args(
        input, footprint, mode, origin, 'footprint', axes=axes)
    if default_footprint:
        filter_size = footprint.size
    else:
        filter_size = int(footprint.sum())  # synchronize
    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return _min_or_max_filter(input, None, footprint, None, output, modes,
                                  cval, origins, 'min', axes)
    if rank == filter_size - 1:
        return _min_or_max_filter(input, None, footprint, None, output, modes,
                                  cval, origins, 'max', axes)
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)
    kernel = _get_rank_kernel(filter_size, rank, modes, footprint.shape,
                              offsets, float(cval), int_type)
    return _filters_core._call_kernel(kernel, input, footprint, output,
                                      weights_dtype=bool)


__SHELL_SORT = '''
__device__ void sort(X *array, int size) {{
    int gap = {gap};
    while (gap > 1) {{
        gap /= 3;
        for (int i = gap; i < size; ++i) {{
            X value = array[i];
            int j = i - gap;
            while (j >= 0 && value < array[j]) {{
                array[j + gap] = array[j];
                j -= gap;
            }}
            array[j + gap] = value;
        }}
    }}
}}'''


@cupy._util.memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3*gap+1
    return gap


@cupy._util.memoize(for_each_device=True)
def _get_rank_kernel(filter_size, rank, modes, w_shape, offsets, cval,
                     int_type):
    s_rank = min(rank, filter_size - rank - 1)
    # The threshold was set based on the measurements on a V100
    # TODO(leofang, anaruse): Use Optuna to automatically tune the threshold,
    # as it may vary depending on the GPU in use, compiler version, dtype,
    # filter size, etc.
    if s_rank <= 80:
        # When s_rank is small and register usage is low, this partial
        # selection sort approach is faster than general sorting approach
        # using shell sort.
        if s_rank == rank:
            comp_op = '<'
        else:
            comp_op = '>'
        array_size = s_rank + 2
        found_post = f'''
            if (iv > {s_rank} + 1) {{{{
                int target_iv = 0;
                X target_val = values[0];
                for (int jv = 1; jv <= {s_rank} + 1; jv++) {{{{
                    if (target_val {comp_op} values[jv]) {{{{
                        target_val = values[jv];
                        target_iv = jv;
                    }}}}
                }}}}
                if (target_iv <= {s_rank}) {{{{
                    values[target_iv] = values[{s_rank} + 1];
                }}}}
                iv = {s_rank} + 1;
            }}}}'''
        post = f'''
            X target_val = values[0];
            for (int jv = 1; jv <= {s_rank}; jv++) {{
                if (target_val {comp_op} values[jv]) {{
                    target_val = values[jv];
                }}
            }}
            y=cast<Y>(target_val);'''
        sorter = ''
    else:
        array_size = filter_size
        found_post = ''
        post = f'sort(values,{filter_size});\ny=cast<Y>(values[{rank}]);'
        sorter = __SHELL_SORT.format(gap=_get_shell_gap(filter_size))

    return _filters_core._generate_nd_kernel(
        f'rank_{filter_size}_{rank}',
        f'int iv = 0;\nX values[{array_size}];',
        'values[iv++] = {value};' + found_post, post,
        modes, w_shape, int_type, offsets, cval, preamble=sorter)


def generic_filter(input, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0, *,
                   axes=None):
    """Compute a multi-dimensional filter using the provided raw kernel or
    reduction kernel.

    Unlike the scipy.ndimage function, this does not support the
    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant
    restrictions on the ``function`` provided.

    Args:
        input (cupy.ndarray): The input array.
        function (cupy.ReductionKernel or cupy.RawKernel):
            The kernel or function to apply to each region.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. note::
        If the `function` is a :class:`cupy.RawKernel` then it must be for a
        function that has the following signature. Unlike most functions, this
        should not utilize `blockDim`/`blockIdx`/`threadIdx`::

            __global__ void func(double *buffer, int filter_size,
                                 double *return_value)

        If the `function` is a :class:`cupy.ReductionKernel` then it must be
        for a kernel that takes 1 array input and produces 1 'scalar' output.

    .. seealso:: :func:`scipy.ndimage.generic_filter`
    """
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    _, footprint, _ = _filters_core._check_size_footprint_structure(
        num_axes, size, footprint, None, 2, True)
    filter_size = int(footprint.sum())
    axes, footprint, origins, _, int_type = _filters_core._check_nd_args(
        input, footprint, mode, origin, 'footprint', axes=axes)
    in_dtype = input.dtype
    sub = _filters_generic._get_sub_kernel(function)
    if footprint.size == 0:
        return cupy.zeros_like(input)
    output = _util._get_output(output, input)
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)
    args = (filter_size, mode, footprint.shape,
            offsets, float(cval), int_type)
    if isinstance(sub, cupy.RawKernel):
        kernel = _filters_generic._get_generic_filter_raw(sub, *args)
    elif isinstance(sub, cupy.ReductionKernel):
        kernel = _filters_generic._get_generic_filter_red(
            sub, in_dtype, output.dtype, *args)
    return _filters_core._call_kernel(kernel, input, footprint, output,
                                      weights_dtype=bool)


def generic_filter1d(input, function, filter_size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Compute a 1D filter along the given axis using the provided raw kernel.

    Unlike the scipy.ndimage function, this does not support the
    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant
    restrictions on the ``function`` provided.

    Args:
        input (cupy.ndarray): The input array.
        function (cupy.RawKernel): The kernel to apply along each axis.
        filter_size (int): Length of the filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. note::
        The provided function (as a RawKernel) must have the following
        signature. Unlike most functions, this should not utilize
        `blockDim`/`blockIdx`/`threadIdx`::

            __global__ void func(double *input_line, ptrdiff_t input_length,
                                 double *output_line, ptrdiff_t output_length)

    .. seealso:: :func:`scipy.ndimage.generic_filter1d`
    """
    # This filter is very different than all other filters (including
    # generic_filter and all 1d filters) and it has a customized solution.
    # It is also likely fairly terrible, but only so much can be done when
    # matching the scipy interface of having the sub-kernel work on entire
    # lines of data.
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    if not isinstance(function, cupy.RawKernel):
        raise TypeError('bad function type')
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    axis = internal._normalize_axis_index(axis, input.ndim)
    origin = _util._check_origin(origin, filter_size)
    _util._check_mode(mode)
    output = _util._get_output(output, input)
    in_ctype = cupy._core._scalar.get_typename(input.dtype)
    out_ctype = cupy._core._scalar.get_typename(output.dtype)
    int_type = _util._get_inttype(input)
    n_lines = input.size // input.shape[axis]
    kernel = _filters_generic._get_generic_filter1d(
        function, input.shape[axis], n_lines, filter_size,
        origin, mode, float(cval), in_ctype, out_ctype, int_type)
    data = cupy.array(
        (axis, input.ndim) + input.shape + input.strides + output.strides,
        dtype=cupy.int32 if int_type == 'int' else cupy.int64)
    kernel(((n_lines+128-1) // 128,), (128,), (input, output, data))
    return output
