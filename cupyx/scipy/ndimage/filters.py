import cupy


def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
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

    Returns:
        cupy.ndarray: The result of correlate.

    .. seealso:: :func:`scipy.ndimage.correlate`
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin)


def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
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

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.convolve`
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin,
                                  True)


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
    """
    weights, origins = _convert_1d_args(input.ndim, weights, origin, axis)
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
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    weights, origins = _convert_1d_args(input.ndim, weights, origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins)


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution=False):
    origins, int_type = _check_nd_args(input, weights, mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    kernel = _get_correlate_kernel(mode, weights.shape, int_type,
                                   origins, cval)
    return _call_kernel(kernel, input, weights, output)


@cupy.util.memoize(for_each_device=True)
def _get_correlate_kernel(mode, wshape, int_type, origins, cval):
    return _generate_nd_kernel(
        'correlate',
        'W sum = (W)0;',
        'sum += cast<W>({value}) * wval;',
        'y = cast<Y>(sum);',
        mode, wshape, int_type, origins, cval, ctype='W')


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
    """
    return correlate1d(input, cupy.ones(size) / size, axis, output, mode, cval,
                       origin)


def uniform_filter(input, size=3, output=None, mode="reflect", cval=0.0,
                   origin=0):
    """Multi-dimensional uniform filter.
    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): Lengths of the uniform filter for each
            dimension. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of ``0`` is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.uniform_filter`
    """
    sizes = _fix_sequence_arg(size, input.ndim, 'size', int)

    def get(size):
        return None if size <= 1 else cupy.ones(size) / size

    return _run_1d_correlates(input, sizes, get, output, mode, cval, origin)


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
    """
    radius = int(float(truncate) * float(sigma) + 0.5)
    weights = _gaussian_kernel1d(sigma, int(order), radius)
    return correlate1d(input, weights, axis, output, mode, cval)


def gaussian_filter(input, sigma, order=0, output=None, mode="reflect",
                    cval=0.0, truncate=4.0):
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
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_filter`
    """
    sigmas = _fix_sequence_arg(sigma, input.ndim, 'sigma', float)
    orders = _fix_sequence_arg(order, input.ndim, 'order', int)
    truncate = float(truncate)

    def get(param):
        sigma, order = param
        radius = int(truncate * float(sigma) + 0.5)
        if radius <= 0:
            return None
        return _gaussian_kernel1d(sigma, order, radius)

    return _run_1d_correlates(input, list(zip(sigmas, orders)), get, output,
                              mode, cval, 0)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian correlation kernel.
    """
    import numpy
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x /= phi_x.sum()

    if order == 0:
        return cupy.asarray(phi_x)

    # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
    # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
    # p'(x) = -1 / sigma ** 2
    # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
    # coefficients of q(x)
    exponent_range = numpy.arange(order + 1)
    q = numpy.zeros(order + 1)
    q[0] = 1
    D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
    P = numpy.diag(numpy.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1])


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
    """
    return _prewitt_or_sobel(input, axis, output, mode, cval, cupy.ones(3))


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
    """
    return _prewitt_or_sobel(input, axis, output, mode, cval,
                             cupy.array([1, 2, 1]))


def _prewitt_or_sobel(input, axis, output, mode, cval, weights):
    axis = cupy.util._normalize_axis_index(axis, input.ndim)

    def get(is_diff):
        return cupy.array([-1, 0, 1]) if is_diff else weights

    return _run_1d_correlates(input, [a == axis for a in range(input.ndim)],
                              get, output, mode, cval)


def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0, extra_arguments=(), extra_keywords=None):
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
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.generic_laplace`
    """
    if extra_keywords is None:
        extra_keywords = {}
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    output = _get_output(output, input)
    if input.ndim == 0:
        output[...] = input[...]
        return output
    derivative2(input, 0, output, modes[0], cval,
                *extra_arguments, **extra_keywords)
    if input.ndim > 1:
        tmp = _get_output(output.dtype, input)
        for i in range(1, input.ndim):
            derivative2(input, i, tmp, modes[i], cval,
                        *extra_arguments, **extra_keywords)
            output += tmp
    return output


def laplace(input, output=None, mode="reflect", cval=0.0):
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
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.laplace`
    """
    weights = cupy.array([1, -2, 1], dtype=cupy.float64)

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, weights, axis, output, mode, cval)

    return generic_laplace(input, derivative2, output, mode, cval)


def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs):
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
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_laplace`
    """
    def derivative2(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)
    return generic_laplace(input, derivative2, output, mode, cval)


def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
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
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.generic_gradient_magnitude`
    """
    if extra_keywords is None:
        extra_keywords = {}
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    output = _get_output(output, input)
    if input.ndim == 0:
        output[...] = input[...]
        return output
    derivative(input, 0, output, modes[0], cval,
               *extra_arguments, **extra_keywords)
    output *= output
    if input.ndim > 1:
        tmp = _get_output(output.dtype, input)
        for i in range(1, input.ndim):
            derivative(input, i, tmp, modes[i], cval,
                       *extra_arguments, **extra_keywords)
            tmp *= tmp
            output += tmp
    return cupy.sqrt(output, output, casting='unsafe')


def gaussian_gradient_magnitude(input, sigma, output=None, mode="reflect",
                                cval=0.0, **kwargs):
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
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_gradient_magnitude`
    """
    def derivative(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)
    return generic_gradient_magnitude(input, derivative, output, mode, cval)


def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
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
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 'min')


def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
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
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 'max')


def _min_or_max_filter(input, size, footprint, structure, output, mode, cval,
                       origin, func):
    # structure is used by morphology.grey_erosion() and grey_dilation()
    # and not by the regular min/max filters

    sizes, footprint, structure = _check_size_footprint_structure(
        input.ndim, size, footprint, structure)

    if sizes is not None:
        # Seperable filter, run as a series of 1D filters
        fltr = minimum_filter1d if func == 'min' else maximum_filter1d
        return _run_1d_filters([fltr if size > 1 else None for size in sizes],
                               input, sizes, output, mode, cval, origin)

    origins, int_type = _check_nd_args(input, footprint, mode, origin,
                                       'footprint')
    if structure is not None and structure.ndim != input.ndim:
        raise RuntimeError('structure array has incorrect shape')

    if footprint.size == 0:
        return cupy.zeros_like(input)
    center = tuple(x//2 + origin
                   for x, origin in zip(footprint.shape, origins))
    kernel = _get_min_or_max_kernel(mode, footprint.shape, func,
                                    origins, float(cval), int_type,
                                    has_structure=structure is not None,
                                    has_central_value=bool(footprint[center]))
    return _call_kernel(kernel, input, footprint, output, structure,
                        weights_dtype=bool)


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
    ftprnt, origins = _convert_1d_args(input.ndim, ftprnt, origin, axis)
    origins, int_type = _check_nd_args(input, ftprnt, mode, origins,
                                       'footprint')
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, origins,
                                    float(cval), int_type, has_weights=False)
    return _call_kernel(kernel, input, None, output, weights_dtype=bool)


@cupy.util.memoize(for_each_device=True)
def _get_min_or_max_kernel(mode, wshape, func, origins, cval, int_type,
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

    return _generate_nd_kernel(
        func, pre.format(ctype),
        found.format(func=func, value=value), 'y = cast<Y>(value);',
        mode, wshape, int_type, origins, cval, ctype=ctype,
        has_weights=has_weights, has_structure=has_structure)


def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0):
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
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = int(rank)
    return _rank_filter(input, lambda fs: rank+fs if rank < 0 else rank,
                        size, footprint, output, mode, cval, origin)


def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
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
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(input, lambda fs: fs//2,
                        size, footprint, output, mode, cval, origin)


def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0):
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
                        size, footprint, output, mode, cval, origin)


def _rank_filter(input, get_rank, size=None, footprint=None, output=None,
                 mode="reflect", cval=0.0, origin=0):
    _, footprint, _ = _check_size_footprint_structure(
        input.ndim, size, footprint, None, force_footprint=True)
    origins, int_type = _check_nd_args(input, footprint, mode, origin,
                                       'footprint')
    if footprint.size == 0:
        return cupy.zeros_like(input)
    filter_size = int(footprint.sum())
    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return _min_or_max_filter(input, None, footprint, None, output, mode,
                                  cval, origins, 'min')
    if rank == filter_size - 1:
        return _min_or_max_filter(input, None, footprint, None, output, mode,
                                  cval, origins, 'max')
    kernel = _get_rank_kernel(filter_size, rank, mode, footprint.shape,
                              origins, float(cval), int_type)
    return _call_kernel(kernel, input, footprint, output, None, bool)


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


__SELECTION_SORT = '''
__device__ void sort(X *array, int size) {
    for (int i = 0; i < size-1; ++i) {
        X min_val = array[i];
        int min_idx = i;
        for (int j = i+1; j < size; ++j) {
            X val_j = array[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx) {
            array[min_idx] = array[i];
            array[i] = min_val;
        }
    }
}'''


@cupy.util.memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3*gap+1
    return gap


@cupy.util.memoize(for_each_device=True)
def _get_rank_kernel(filter_size, rank, mode, wshape, origins, cval, int_type):
    # Below 225 (15x15 median filter) selection sort is 1.5-2.5x faster
    # Above, shell sort does progressively better (by 3025 (55x55) it is 9x)
    # Also tried insertion sort, which is always slower than either one
    sorter = __SELECTION_SORT if filter_size <= 255 else \
        __SHELL_SORT.format(gap=_get_shell_gap(filter_size))
    return _generate_nd_kernel(
        'rank_{}_{}'.format(filter_size, rank),
        'int iv = 0;\nX values[{}];'.format(filter_size),
        'values[iv++] = {value};',
        'sort(values,{});\ny=cast<Y>(values[{}]);'.format(filter_size, rank),
        mode, wshape, int_type, origins, cval, preamble=sorter)


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if isinstance(output, cupy.ndarray):
        if output.shape != tuple(shape):
            raise ValueError('output shape is not correct')
    else:
        dtype = output
        if dtype is None:
            dtype = input.dtype
        output = cupy.zeros(shape, dtype)
    return output


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if isinstance(arg, str):
        return [conv(arg)] * ndim
    try:
        arg = iter(arg)
    except TypeError:
        return [conv(arg)] * ndim
    lst = [conv(x) for x in arg]
    if len(lst) != ndim:
        msg = "{} must have length equal to input rank".format(name)
        raise RuntimeError(msg)
    return lst


def _check_origin(origin, width):
    origin = int(origin)
    if (width // 2 + origin < 0) or (width // 2 + origin >= width):
        raise ValueError('invalid origin')
    return origin


def _check_mode(mode):
    if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
        msg = 'boundary mode not supported (actual: {})'.format(mode)
        raise RuntimeError(msg)
    return mode


def _check_size_footprint_structure(ndim, size, footprint, structure,
                                    stacklevel=3, force_footprint=False):
    if structure is None and footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _fix_sequence_arg(size, ndim, 'size', int)
        if force_footprint:
            return None, cupy.ones(sizes, bool), None
        return sizes, None, None
    if size is not None:
        import warnings
        warnings.warn("ignoring size because {} is set".format(
            'structure' if footprint is None else 'footprint'),
            UserWarning, stacklevel=stacklevel+1)

    if footprint is not None:
        footprint = cupy.array(footprint, bool, True, 'C')
        if not footprint.any():
            raise ValueError("all-zero footprint is not supported")

    if structure is None:
        if not force_footprint and footprint.all():
            if footprint.ndim != ndim:
                raise RuntimeError("size must have length equal to input rank")
            return footprint.shape, None, None
        return None, footprint, None

    structure = cupy.ascontiguousarray(structure)
    if footprint is None:
        footprint = cupy.ones(structure.shape, bool)
    return None, footprint, structure


def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError('incorrect filter size')
    axis = cupy.util._normalize_axis_index(axis, ndim)
    wshape = [1]*ndim
    wshape[axis] = weights.size
    weights = weights.reshape(wshape)
    origins = [0]*ndim
    origins[axis] = _check_origin(origin, weights.size)
    return weights, tuple(origins)


def _check_nd_args(input, weights, mode, origin, wghts_name='filter weights'):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    _check_mode(mode)
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = sum((x-1)*abs(stride) for x, stride in
                 zip(input.shape, input.strides)) + input.dtype.itemsize
    int_type = 'int' if nbytes < (1 << 31) else 'ptrdiff_t'
    # However, weights must always be 2 GiB or less
    if weights.nbytes > (1 << 31):
        raise RuntimeError('weights must be 2 GiB or less, use FFTs instead')
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _fix_sequence_arg(origin, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _check_origin(origin, width)
    return tuple(origins), int_type


def _run_1d_filters(filters, input, args, output, mode, cval, origin=0):
    """
    Runs a series of 1D filters forming an nd filter. The filters must be a
    list of callables that take input, arg, axis, output, mode, cval, origin.
    The args is a list of values that are passed for the arg value to the
    filter. Individual filters can be None causing that axis to be skipped.
    """
    output_orig = output
    output = _get_output(output, input)
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    origins = _fix_sequence_arg(origin, input.ndim, 'origin', int)
    n_filters = sum(filter is not None for filter in filters)
    if n_filters == 0:
        output[...] = input[...]
        return output
    # We can't operate in-place efficiently, so use a 2-buffer system
    temp = _get_output(output.dtype, input) if n_filters > 1 else None
    first = True
    iterator = zip(filters, args, modes, origins)
    for axis, (fltr, arg, mode, origin) in enumerate(iterator):
        if fltr is None:
            continue
        fltr(input, arg, axis, output, mode, cval, origin)
        input, output = output, temp if first else input
    if isinstance(output_orig, cupy.ndarray) and input is not output_orig:
        output_orig[...] = input
        input = output_orig
    return input


def _run_1d_correlates(input, params, get_weights, output, mode, cval,
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
    return _run_1d_filters([None if w is None else correlate1d for w in wghts],
                           input, wghts, output, mode, cval, origin)


def _call_kernel(kernel, input, weights, output, structure=None,
                 weights_dtype=cupy.float64, structure_dtype=cupy.float64):
    """
    Calls a constructed ElementwiseKernel. The kernel must take an input image,
    an optional array of weights, an optional array for the structure, and an
    output array.

    weights and structure can be given as None (structure defaults to None) in
    which case they are not passed to the kernel at all. If the output is given
    as None then it will be allocated in this function.

    This function deals with making sure that the weights and structure are
    contiguous and float64 (or bool for weights that are footprints)*, that the
    output is allocated and appriopately shaped. This also deals with the
    situation that the input and output arrays overlap in memory.

    * weights is always cast to float64 or bool in order to get an output
    compatible with SciPy, though float32 might be sufficient when input dtype
    is low precision. If weights_dtype is passed as weights.dtype then no
    dtype conversion will occur. The input and output are never converted.
    """
    args = [input]
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weights_dtype)
        args.append(weights)
    if structure is not None:
        structure = cupy.ascontiguousarray(structure, structure_dtype)
        args.append(structure)
    output = _get_output(output, input)
    needs_temp = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = _get_output(output.dtype, input), output
    args.append(output)
    kernel(*args)
    if needs_temp:
        temp[...] = output[...]
        output = temp
    return output


def _generate_boundary_condition_ops(mode, ix, xsize):
    if mode == 'reflect':
        ops = '''
        if ({ix} < 0) {{
            {ix} = - 1 - {ix};
        }}
        {ix} %= {xsize} * 2;
        {ix} = min({ix}, 2 * {xsize} - 1 - {ix});'''.format(ix=ix, xsize=xsize)
    elif mode == 'mirror':
        ops = '''
        if ({ix} < 0) {{
            {ix} = - {ix};
        }}
        if ({xsize} == 1) {{
            {ix} = 0;
        }} else {{
            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
            {ix} = min({ix}, 2 * {xsize} - 2 - {ix});
        }}'''.format(ix=ix, xsize=xsize)
    elif mode == 'nearest':
        ops = '''
        {ix} = min(max({ix}, 0), {xsize} - 1);'''.format(ix=ix, xsize=xsize)
    elif mode == 'wrap':
        ops = '''
        if ({ix} < 0) {{
            {ix} += (1 - ({ix} / {xsize})) * {xsize};
        }}
        {ix} %= {xsize};'''.format(ix=ix, xsize=xsize)
    elif mode == 'constant':
        ops = '''
        if ({ix} >= {xsize}) {{
            {ix} = -1;
        }}'''.format(ix=ix, xsize=xsize)
    return ops


_CAST_FUNCTION = """
// Implements a casting function to make it compatible with scipy
// Use like cast<to_type>(value)
// It's actually really simple - most of this is <type_traits>

// Small bit of <type_traits> which cannot be imported in NVRTC
// Requires compiling with --std=c++11 or higher
template<bool B, class T=void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };
template<class T> struct remove_const          { typedef T type; };
template<class T> struct remove_const<const T> { typedef T type; };
template<class T> struct remove_volatile             { typedef T type; };
template<class T> struct remove_volatile<volatile T> { typedef T type; };
template<class T> struct remove_cv {
  typedef typename remove_volatile<typename remove_const<T>::type>::type type;
};
template<class T, T v>
struct integral_constant { static constexpr T value = v; };
//struct true_type { static constexpr bool value = true; };
//struct false_type { static constexpr bool value = false; };
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;
template<class T> struct __is_fp : public false_type {};
template<>        struct __is_fp<float> : public true_type {};
template<>        struct __is_fp<double> : public true_type {};
template<>        struct __is_fp<long double> : public true_type {};
template<class T> struct is_floating_point
    : public __is_fp<typename remove_cv<T>::type> {};
template<class T> struct is_signed : integral_constant<bool, (T)(-1)<0> {};

template <typename B, typename A>
__device__
typename enable_if<!is_floating_point<A>::value||is_signed<B>::value, B>::type
cast(A a) { return (B)a; }

template <typename B, typename A>
__device__
typename enable_if<is_floating_point<A>::value&&!is_signed<B>::value, B>::type
cast(A a) { return (a >= 0) ? (B)a : -(B)(-a); }


"""


def _generate_nd_kernel(name, pre, found, post, mode, wshape, int_type,
                        origins, cval, ctype='X', preamble='', options=(),
                        has_weights=True, has_structure=False):
    # Currently this code uses CArray for weights but avoids using CArray for
    # the input data and instead does the indexing itself since it is faster.
    # If CArray becomes faster than follow the comments that start with
    # CArray: to switch over to using CArray for the input data as well.

    ndim = len(wshape)
    in_params = 'raw X x'
    if has_weights:
        in_params += ', raw W w'
    if has_structure:
        in_params += ', raw S s'
    out_params = 'Y y'

    inds = _generate_indices_ops(
        ndim, int_type, 'xsize_{j}',
        [' - {}'.format(wshape[j]//2 + origins[j]) for j in range(ndim)])
    # CArray: remove xstride_{j}=... from string
    sizes = ['{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];'.
             format(j=j, type=int_type) for j in range(ndim)]
    # CArray: remove expr entirely
    expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])

    ws_init = ws_pre = ws_post = ''
    if has_weights or has_structure:
        ws_init = 'int iws = 0;'
        if has_structure:
            ws_pre = 'S sval = s[iws];\n'
        if has_weights:
            ws_pre += 'W wval = w[iws];\nif (wval)'
        ws_post = 'iws++;'

    loops = []
    for j in range(ndim):
        if wshape[j] == 1:
            # CArray: string becomes 'inds[{j}] = ind_{j};', remove (int_)type
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.
                         format(j=j, type=int_type))
        else:
            boundary = _generate_boundary_condition_ops(
                mode, 'ix_{}'.format(j), 'xsize_{}'.format(j))
            # CArray: last line of string becomes inds[{j}] = ix_{j};
            loops.append('''
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        '''.format(j=j, wsize=wshape[j], boundary=boundary, type=int_type))

    # CArray: string becomes 'x[inds]', no format call needed
    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
        value = '(({cond}) ? cast<{ctype}>({cval}) : {value})'.format(
            cond=cond, ctype=ctype, cval=cval, value=value)
    found = found.format(value=value)

    # CArray: replace comment and next line in string with
    #   {type} inds[{ndim}] = {{0}};
    # and add ndim=ndim, type=int_type to format call
    operation = '''
    {sizes}
    {inds}
    // don't use a CArray for indexing (faster to deal with indexing ourselves)
    const unsigned char* data = (const unsigned char*)&x[0];
    {ws_init}
    {pre}
    {loops}
        // inner-most loop
        {ws_pre} {{
            {found}
        }}
        {ws_post}
    {end_loops}
    {post}
    '''.format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post,
               ws_init=ws_init, ws_pre=ws_pre, ws_post=ws_post,
               loops='\n'.join(loops), found=found, end_loops='}'*ndim)

    name = 'cupy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode, '_'.join(['{}'.format(j) for j in wshape]))
    if int_type == 'ptrdiff_t':
        name += '_i64'
    if has_structure:
        name += '_with_structure'
    preamble = _CAST_FUNCTION + preamble
    options += ('--std=c++11',)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)


def _generate_indices_ops(ndim, int_type, xsize='x.shape()[{j}]', extras=None):
    if extras is None:
        extras = ('',)*ndim
    code = '{type} ind_{j} = _i % ' + xsize + '{extra}; _i /= ' + xsize + ';'
    body = [code.format(type=int_type, j=j, extra=extras[j])
            for j in range(ndim-1, 0, -1)]
    return '{type} _i = i;\n{body}\n{type} ind_0 = _i{extra};'.format(
        type=int_type, body='\n'.join(body), extra=extras[0])
