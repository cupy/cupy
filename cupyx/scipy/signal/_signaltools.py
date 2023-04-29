import warnings

import cupy
from cupy._core import internal
from cupy.linalg import lstsq

from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
from cupyx.scipy.signal._iir_utils import apply_iir, compute_correction_factors
from cupyx.scipy.signal._arraytools import axis_slice, axis_assign


def convolve(in1, in2, mode='full', method='auto'):
    """Convolve two N-dimensional arrays.

    Convolve ``in1`` and ``in2``, with the output size determined by the
    ``mode`` argument.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as `in1`.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        method (str): Indicates which method to use for the computations:

            - ``'direct'``: The convolution is determined directly from sums, \
                the definition of convolution
            - ``'fft'``: The Fourier Transform is used to perform the \
                convolution by calling ``fftconvolve``.
            - ``'auto'``: Automatically choose direct of FFT based on an \
                estimate of which is faster for the arguments (default).

    Returns:
        cupy.ndarray: the result of convolution.

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.correlation`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.convolve`
    .. note::
        By default, ``convolve`` and ``correlate`` use ``method='auto'``, which
        calls ``choose_conv_method`` to choose the fastest method using
        pre-computed values. CuPy may not choose the same method to compute
        the convolution as SciPy does given the same inputs.
    """
    return _correlate(in1, in2, mode, method, True)


def correlate(in1, in2, mode='full', method='auto'):
    """Cross-correlate two N-dimensional arrays.

    Cross-correlate ``in1`` and ``in2``, with the output size determined by the
    ``mode`` argument.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        method (str): Indicates which method to use for the computations:

            - ``'direct'``: The convolution is determined directly from sums, \
                the definition of convolution
            - ``'fft'``: The Fourier Transform is used to perform the \
                convolution by calling ``fftconvolve``.
            - ``'auto'``: Automatically choose direct of FFT based on an \
                estimate of which is faster for the arguments (default).

    Returns:
        cupy.ndarray: the result of correlation.

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.correlation`
    .. seealso:: :func:`scipy.signal.correlation`
    .. note::
        By default, ``convolve`` and ``correlate`` use ``method='auto'``, which
        calls ``choose_conv_method`` to choose the fastest method using
        pre-computed values. CuPy may not choose the same method to compute
        the convolution as SciPy does given the same inputs.
    """
    return _correlate(in1, in2, mode, method, False)


def _correlate(in1, in2, mode='full', method='auto', convolution=False):
    quick_out = _st_core._check_conv_inputs(in1, in2, mode, convolution)
    if quick_out is not None:
        return quick_out
    if method not in ('auto', 'direct', 'fft'):
        raise ValueError('acceptable methods are "auto", "direct", or "fft"')

    if method == 'auto':
        method = choose_conv_method(in1, in2, mode=mode)

    if method == 'direct':
        return _st_core._direct_correlate(in1, in2, mode, in1.dtype,
                                          convolution)

    # if method == 'fft':
    if not convolution:
        in2 = _st_core._reverse(in2).conj()
    inputs_swapped = _st_core._inputs_swap_needed(mode, in1.shape, in2.shape)
    if inputs_swapped:
        in1, in2 = in2, in1
    out = fftconvolve(in1, in2, mode)
    result_type = cupy.result_type(in1, in2)
    if result_type.kind in 'ui':
        out = out.round()
    out = out.astype(result_type, copy=False)
    return out


def fftconvolve(in1, in2, mode='full', axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve ``in1`` and ``in2`` using the fast Fourier transform method, with
    the output size determined by the ``mode`` argument.

    This is generally much faster than the ``'direct'`` method of ``convolve``
    for large arrays, but can be slower when only a few output values are
    needed, and can only output float arrays (int or object array inputs will
    be cast to float).

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear \
                          cross-correlation (default)
            - ``'valid'``: output consists only of those elements that do \
                           not rely on the zero-padding. Either ``in1`` or \
                           ``in2`` must be at least as large as the other in \
                           every dimension.
            - ``'same'``: output is the same size as ``in1``, centered \
                          with respect to the 'full' output

        axes (scalar or tuple of scalar or None): Axes over which to compute
            the convolution. The default is over all axes.

    Returns:
        cupy.ndarray: the result of convolution

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.correlation`
    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.correlation`
    """
    out = _st_core._check_conv_inputs(in1, in2, mode)
    if out is not None:
        return out
    in1, in2, axes = _st_core._init_freq_conv_axes(in1, in2, mode, axes, False)
    shape = [max(x1, x2) if a not in axes else x1 + x2 - 1
             for a, (x1, x2) in enumerate(zip(in1.shape, in2.shape))]
    out = _st_core._freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)
    return _st_core._apply_conv_mode(out, in1.shape, in2.shape, mode, axes)


def choose_conv_method(in1, in2, mode='full'):
    """Find the fastest convolution/correlation method.

    Args:
        in1 (cupy.ndarray): first input.
        in2 (cupy.ndarray): second input.
        mode (str, optional): ``'valid'``, ``'same'``, ``'full'``.

    Returns:
        str: A string indicating which convolution method is fastest,
        either ``'direct'`` or ``'fft'``.

    .. warning::
        This function currently doesn't support measure option,
        nor multidimensional inputs. It does not guarantee
        the compatibility of the return value to SciPy's one.

    .. seealso:: :func:`scipy.signal.choose_conv_method`

    """
    return cupy._math.misc._choose_conv_method(in1, in2, mode)


def oaconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using the overlap-add method.

    Convolve ``in1`` and ``in2`` using the overlap-add method, with the output
    size determined by the ``mode`` argument. This is generally faster than
    ``convolve`` for large arrays, and generally faster than ``fftconvolve``
    when one array is much larger than the other, but can be slower when only a
    few output values are needed or when the arrays are very similar in shape,
    and can only output float arrays (int or object array inputs will be cast
    to float).

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear \
                          cross-correlation (default)
            - ``'valid'``: output consists only of those elements that do \
                           not rely on the zero-padding. Either ``in1`` or \
                           ``in2`` must be at least as large as the other in \
                           every dimension.
            - ``'same'``: output is the same size as ``in1``, centered \
                          with respect to the ``'full'`` output

        axes (scalar or tuple of scalar or None): Axes over which to compute
            the convolution. The default is over all axes.

    Returns:
        cupy.ndarray: the result of convolution

    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.oaconvolve`
    """
    out = _st_core._check_conv_inputs(in1, in2, mode)
    if out is not None:
        return out
    if in1.shape == in2.shape:  # Equivalent to fftconvolve
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    in1, in2, axes = _st_core._init_freq_conv_axes(in1, in2, mode, axes,
                                                   sorted_axes=True)
    s1, s2 = in1.shape, in2.shape
    if not axes:
        return _st_core._apply_conv_mode(in1*in2, s1, s2, mode, axes)

    # Calculate the block sizes for the output, steps, first and second inputs.
    # It is simpler to calculate them all together than doing them in separate
    # loops due to all the special cases that need to be handled.
    optimal_sizes = (_st_core._calc_oa_lens(s1[i], s2[i]) if i in axes else
                     (-1, -1, s1[i], s2[i]) for i in range(in1.ndim))
    block_size, overlaps, in1_step, in2_step = zip(*optimal_sizes)

    # Fall back to fftconvolve if there is only one block in every dimension
    if in1_step == s1 and in2_step == s2:
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    # Pad and reshape the inputs for overlapping and adding
    shape_final = [s1[i]+s2[i]-1 if i in axes else None
                   for i in range(in1.ndim)]
    in1, in2 = _st_core._oa_reshape_inputs(in1, in2, axes, shape_final,
                                           block_size, overlaps,
                                           in1_step, in2_step)

    # Reshape the overlap-add parts to input block sizes
    split_axes = [iax+i for i, iax in enumerate(axes)]
    fft_axes = [iax+1 for iax in split_axes]

    # Do the convolution
    fft_shape = [block_size[i] for i in axes]
    ret = _st_core._freq_domain_conv(in1, in2, fft_axes, fft_shape,
                                     calc_fast_len=False)

    # Do the overlap-add
    for ax, ax_fft, ax_split in zip(axes, fft_axes, split_axes):
        overlap = overlaps[ax]
        if overlap is None:
            continue

        ret, overpart = cupy.split(ret, [-overlap], ax_fft)
        overpart = cupy.split(overpart, [-1], ax_split)[0]

        ret_overpart = cupy.split(ret, [overlap], ax_fft)[0]
        ret_overpart = cupy.split(ret_overpart, [1], ax_split)[1]
        ret_overpart += overpart

    # Reshape back to the correct dimensionality
    shape_ret = [ret.shape[i] if i not in fft_axes else
                 ret.shape[i]*ret.shape[i-1]
                 for i in range(ret.ndim) if i not in split_axes]
    ret = ret.reshape(*shape_ret)

    # Slice to the correct size
    ret = ret[tuple([slice(islice) for islice in shape_final])]

    return _st_core._apply_conv_mode(ret, s1, s2, mode, axes)


def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """Convolve two 2-dimensional arrays.

    Convolve ``in1`` and ``in2`` with output size determined by ``mode``, and
    boundary conditions determined by ``boundary`` and ``fillvalue``.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        boundary (str): Indicates how to handle boundaries:

            - ``fill``: pad input arrays with fillvalue (default)
            - ``wrap``: circular boundary conditions
            - ``symm``: symmetrical boundary conditions

        fillvalue (scalar): Value to fill pad input arrays with. Default is 0.

    Returns:
        cupy.ndarray: A 2-dimensional array containing a subset of the discrete
        linear convolution of ``in1`` with ``in2``.

    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.correlate2d`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.convolve2d`
    """
    return _correlate2d(in1, in2, mode, boundary, fillvalue, True)


def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """Cross-correlate two 2-dimensional arrays.

    Cross correlate ``in1`` and ``in2`` with output size determined by
    ``mode``, and boundary conditions determined by ``boundary`` and
    ``fillvalue``.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        boundary (str): Indicates how to handle boundaries:

            - ``fill``: pad input arrays with fillvalue (default)
            - ``wrap``: circular boundary conditions
            - ``symm``: symmetrical boundary conditions

        fillvalue (scalar): Value to fill pad input arrays with. Default is 0.

    Returns:
        cupy.ndarray: A 2-dimensional array containing a subset of the discrete
        linear cross-correlation of ``in1`` with ``in2``.

    Note:
        When using ``"same"`` mode with even-length inputs, the outputs of
        ``correlate`` and ``correlate2d`` differ: There is a 1-index offset
        between them.

    .. seealso:: :func:`cupyx.scipy.signal.correlate`
    .. seealso:: :func:`cupyx.scipy.signal.convolve2d`
    .. seealso:: :func:`cupyx.scipy.ndimage.correlate`
    .. seealso:: :func:`scipy.signal.correlate2d`
    """
    return _correlate2d(in1, in2, mode, boundary, fillvalue, False)


def _correlate2d(in1, in2, mode, boundary, fillvalue, convolution=False):
    if not (in1.ndim == in2.ndim == 2):
        raise ValueError('{} inputs must both be 2-D arrays'.format(
            'convolve2d' if convolution else 'correlate2d'))
    _boundaries = {
        'fill': 'constant', 'pad': 'constant',
        'wrap': 'wrap', 'circular': 'wrap',
        'symm': 'reflect', 'symmetric': 'reflect',
    }
    boundary = _boundaries.get(boundary)
    if boundary is None:
        raise ValueError('Acceptable boundary flags are "fill" (or "pad"), '
                         '"circular" (or "wrap"), and '
                         '"symmetric" (or "symm").')
    quick_out = _st_core._check_conv_inputs(in1, in2, mode, convolution)
    if quick_out is not None:
        return quick_out
    return _st_core._direct_correlate(in1, in2, mode, in1.dtype, convolution,
                                      boundary, fillvalue, not convolution)


def wiener(im, mysize=None, noise=None):
    """Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Args:
        im (cupy.ndarray): An N-dimensional array.
        mysize (int or cupy.ndarray, optional): A scalar or an N-length list
            giving the size of the Wiener filter window in each dimension.
            Elements of mysize should be odd. If mysize is a scalar, then this
            scalar is used as the size in each dimension.
        noise (float, optional): The noise-power to use. If None, then noise is
            estimated as the average of the local variance of the input.

    Returns:
        cupy.ndarray: Wiener filtered result with the same shape as `im`.

    .. seealso:: :func:`scipy.signal.wiener`
    """
    if mysize is None:
        mysize = 3
    mysize = _util._fix_sequence_arg(mysize, im.ndim, 'mysize', int)
    im = im.astype(cupy.complex128 if im.dtype.kind == 'c' else cupy.float64,
                   copy=False)

    # Estimate the local mean
    local_mean = _filters.uniform_filter(im, mysize, mode='constant')

    # Estimate the local variance
    local_var = _filters.uniform_filter(im*im, mysize, mode='constant')
    local_var -= local_mean*local_mean

    # Estimate the noise power if needed.
    if noise is None:
        noise = local_var.mean()

    # Perform the filtering
    res = im - local_mean
    res *= 1 - noise / local_var
    res += local_mean
    return cupy.where(local_var < noise, local_mean, res)


def order_filter(a, domain, rank):
    """Perform an order filter on an N-D array.

    Perform an order filter on the array in. The domain argument acts as a mask
    centered over each pixel. The non-zero elements of domain are used to
    select elements surrounding each input pixel which are placed in a list.
    The list is sorted, and the output for that pixel is the element
    corresponding to rank in the sorted list.

    Args:
        a (cupy.ndarray): The N-dimensional input array.
        domain (cupy.ndarray): A mask array with the same number of dimensions
            as `a`. Each dimension should have an odd number of elements.
        rank (int): A non-negative integer which selects the element from the
            sorted list (0 corresponds to the smallest element).

    Returns:
        cupy.ndarray: The results of the order filter in an array with the same
        shape as `a`.

    .. seealso:: :func:`cupyx.scipy.ndimage.rank_filter`
    .. seealso:: :func:`scipy.signal.order_filter`
    """
    if a.dtype.kind in 'bc' or a.dtype == cupy.float16:
        # scipy doesn't support these types
        raise ValueError("data type not supported")
    if any(x % 2 != 1 for x in domain.shape):
        raise ValueError("Each dimension of domain argument "
                         " should have an odd number of elements.")
    return _filters.rank_filter(a, rank, footprint=domain, mode='constant')


def medfilt(volume, kernel_size=None):
    """Perform a median filter on an N-dimensional array.

    Apply a median filter to the input array using a local window-size
    given by `kernel_size`. The array will automatically be zero-padded.

    Args:
        volume (cupy.ndarray): An N-dimensional input array.
        kernel_size (int or list of ints): Gives the size of the median filter
            window in each dimension. Elements of `kernel_size` should be odd.
            If `kernel_size` is a scalar, then this scalar is used as the size
            in each dimension. Default size is 3 for each dimension.

    Returns:
        cupy.ndarray: An array the same size as input containing the median
        filtered result.

    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`scipy.signal.medfilt`
    """
    if volume.dtype.kind == 'c':
        # scipy doesn't support complex
        raise ValueError("complex types not supported")
    if volume.dtype.char == 'e':
        # scipy doesn't support float16
        raise ValueError("float16 type not supported")
    if volume.dtype.kind == 'b':
        # scipy doesn't support bool
        raise ValueError("bool type not supported")
    kernel_size = _get_kernel_size(kernel_size, volume.ndim)
    if any(k > s for k, s in zip(kernel_size, volume.shape)):
        warnings.warn('kernel_size exceeds volume extent: '
                      'volume will be zero-padded')

    size = internal.prod(kernel_size)
    return _filters.rank_filter(volume, size // 2, size=kernel_size,
                                mode='constant')


def medfilt2d(input, kernel_size=3):
    """Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size given
    by `kernel_size` (must be odd). The array is zero-padded automatically.

    Args:
        input (cupy.ndarray): A 2-dimensional input array.
        kernel_size (int of list of ints of length 2): Gives the size of the
            median filter window in each dimension. Elements of `kernel_size`
            should be odd. If `kernel_size` is a scalar, then this scalar is
            used as the size in each dimension. Default is a kernel of size
            (3, 3).

    Returns:
        cupy.ndarray: An array the same size as input containing the median
        filtered result.

    See also
    --------
    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`cupyx.scipy.signal.medfilt`
    .. seealso:: :func:`scipy.signal.medfilt2d`
    """
    if input.dtype.kind == 'c':
        # scipy doesn't support complex
        raise ValueError("complex types not supported")
    if input.dtype.char == 'e':
        # scipy doesn't support float16
        raise ValueError("float16 type not supported")
    if input.dtype.kind == 'b':
        # scipy doesn't support bool
        raise ValueError("bool type not supported")
    if input.ndim != 2:
        raise ValueError('input must be 2d')
    kernel_size = _get_kernel_size(kernel_size, input.ndim)
    order = kernel_size[0] * kernel_size[1] // 2
    return _filters.rank_filter(
        input, order, size=kernel_size, mode='constant')


def lfilter(b, a, x, axis=-1, zi=None):
    """
    Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence, `x`, using a digital filter.  This works for many
    fundamental data types (including Object type).  The filter is a direct
    form II transposed implementation of the standard difference equation
    (see Notes).

    The function `sosfilt` (and filter design using ``output='sos'``) should be
    preferred over `lfilter` for most filtering tasks, as second-order sections
    have fewer numerical problems.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``len(b) + len(a) - 2``. The first ``len(b)`` numbers correspond to the
        last elements of the previous input and the last ``len(a)`` to the last
        elements of the previous output. If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

        **Note**: This argument differs from dimensions from the SciPy
        implementation! However, as long as they are chained from the same
        library, the output result will be the same. Please make sure to use
        the `zi` from CuPy calls and not from SciPy. This due to the parallel
        nature of this implementation as opposed to the serial one in SciPy.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    See Also
    --------
    lfiltic : Construct initial conditions for `lfilter`.
    lfilter_zi : Compute initial state (steady state of step response) for
                 `lfilter`.
    filtfilt : A forward-backward filter, to obtain a filter with zero phase.
    savgol_filter : A Savitzky-Golay filter.
    sosfilt: Filter data using cascaded second-order sections.
    sosfiltfilt: A forward-backward filter using second-order sections.

    Notes
    -----
    The filter function is implemented as a direct II transposed structure.
    This means that the filter implements::

          a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                                - a[1]*y[n-1] - ... - a[N]*y[n-N]

    where `M` is the degree of the numerator, `N` is the degree of the
    denominator, `n` is the sample number and `L` denotes the length of the
    input.  It is implemented by computing first the FIR part and then
    computing the IIR part from it::

             a[0] * y = r(f(x, b), a)
             f(x, b)[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
             r(y, a)[n] = - a[1]*y[n-1] - ... - a[N]*y[n-N]

    The IIR result is computed in parallel by first dividing the input signal
    into chunks (`g_i`) of size `m`. For each chunk, the IIR recurrence
    equation is applied to each chunk (in parallel). Then the chunks are merged
    based on the last N values of the last chunk::

             nc = L/m
             x = [g_0, g_1, ..., g_nc]

             g_i = [x[i * m], ..., x[i * m + m - 1]]
             p_i = r(g_i, a)

             o_i = r(p_i, c(p_{i - 1}))   if i > 1,
                   r(p_i, zi)             otherwise

             y = [o_0, o_1, ..., o_nc]

    where `c` denotes a function that takes a chunk, slices the last `N` values
    and adjust them using a correction factor table computed using the
    (1, 2, ..., N)-fibonacci sequence. For more information see [1]_.

    The rational transfer function describing this filter in the
    z-transform domain is::

                             -1              -M
                 b[0] + b[1]z  + ... + b[M] z
         Y(z) = -------------------------------- X(z)
                             -1              -N
                 a[0] + a[1]z  + ... + a[N] z

    References
    ----------
    .. [1] Sepideh Maleki and Martin Burtscher.
           2018. Automatic Hierarchical Parallelization of Linear Recurrences.
           SIGPLAN Not. 53, 2 (February 2018), 128-138.
           `10.1145/3173162.3173168 <https://doi.org/10.1145/3173162.3173168>`_
    """
    a0 = a[0]
    a_r = - a[1:] / a0
    b = b / a0

    num_b = b.size - 1
    num_a = a_r.size
    x_ndim = x.ndim
    axis = internal._normalize_axis_index(axis, x_ndim)
    n = x.shape[axis]
    fir_dtype = cupy.result_type(x, b)

    prev_in = None
    prev_out = None
    pad_shape = list(x.shape)
    pad_shape[axis] += num_b

    x_full = cupy.zeros(pad_shape, dtype=x.dtype)
    if zi is not None:
        zi = cupy.atleast_1d(zi)
        if num_b > 0:
            prev_in = axis_slice(zi, 0, num_b, axis=axis)
        if num_a > 0:
            prev_out = axis_slice(
                zi, zi.shape[axis] - num_a, zi.shape[axis], axis=axis)

    if prev_in is not None:
        x_full = axis_assign(x_full, prev_in, 0, num_b, axis=axis)

    x_full = axis_assign(x_full, x, num_b, axis=axis)
    origin = -num_b // 2
    out = cupy.empty_like(x_full, dtype=fir_dtype)
    out = _filters.convolve1d(
        x_full, b, axis=axis, mode='constant', origin=origin, output=out)

    if num_b > 0:
        out = axis_slice(out, out.shape[axis] - n, out.shape[axis], axis=axis)

    if a_r.size > 0:
        iir_dtype = cupy.result_type(fir_dtype, a)
        const_dtype = cupy.dtype(a.dtype)
        if const_dtype.kind == 'u':
            const_dtype = cupy.dtype(const_dtype.char.lower())
            a = a.astype(const_dtype)

        out = apply_iir(out, a_r, axis=axis, zi=prev_out, dtype=iir_dtype)

    if zi is not None:
        zi = cupy.empty(zi.shape, dtype=out.dtype)
        if num_b > 0:
            prev_in = axis_slice(
                x, x.shape[axis] - num_b, x.shape[axis], axis=axis)
            zi = axis_assign(zi, prev_in, 0, num_b, axis=axis)
        if num_a > 0:
            prev_out = axis_slice(
                out, out.shape[axis] - num_a, out.shape[axis], axis=axis)
            zi = axis_assign(
                zi, prev_out, zi.shape[axis] - num_a, zi.shape[axis],
                axis=axis)
        return out, zi
    else:
        return out


def lfiltic(b, a, y, x=None):
    """
    Construct initial conditions for lfilter given input and output vectors.

    Given a linear filter (b, a) and initial conditions on the output `y`
    and the input `x`, return the initial conditions on the state vector zi
    which is used by `lfilter` to generate the output given the input.

    Parameters
    ----------
    b : array_like
        Linear filter term.
    a : array_like
        Linear filter term.
    y : array_like
        Initial conditions.
        If ``N = len(a) - 1``, then ``y = {y[-1], y[-2], ..., y[-N]}``.
        If `y` is too short, it is padded with zeros.
    x : array_like, optional
        Initial conditions.
        If ``M = len(b) - 1``, then ``x = {x[-1], x[-2], ..., x[-M]}``.
        If `x` is not given, its initial conditions are assumed zero.
        If `x` is too short, it is padded with zeros.
    axis: int, optional
        The axis to take the initial conditions from, if `x` and `y` are
        n-dimensional

    Returns
    -------
    zi : ndarray
        The state vector ``zi = {z_0[-1], z_1[-1], ..., z_K-1[-1]}``,
        where ``K = M + N``.

    See Also
    --------
    lfilter, lfilter_zi
    """
    # SciPy implementation only supports 1D initial conditions, however,
    # lfilter accepts n-dimensional initial conditions. If SciPy implementation
    # accepts n-dimensional arrays, then axis can be moved to the signature.
    axis = -1
    fir_len = b.size - 1
    iir_len = a.size - 1

    if y is None and x is None:
        return None

    ref_ndim = y.ndim if y is not None else x.ndim
    axis = internal._normalize_axis_index(axis, ref_ndim)

    zi = cupy.empty(0)
    if y is not None and iir_len > 0:
        pad_y = cupy.concatenate(
            (y, cupy.zeros(max(iir_len - y.shape[axis], 0))), axis=axis)
        zi = cupy.take(pad_y, list(range(iir_len)), axis=axis)
        zi = cupy.flip(zi, axis)

    if x is not None and fir_len > 0:
        pad_x = cupy.concatenate(
            (x, cupy.zeros(max(fir_len - x.shape[axis], 0))), axis=axis)
        fir_zi = cupy.take(pad_x, list(range(fir_len)), axis=axis)
        fir_zi = cupy.flip(fir_zi, axis)
        zi = cupy.concatenate((fir_zi, zi), axis=axis)
    return zi


def lfilter_zi(b, a):
    """
    Construct initial conditions for lfilter for step response steady-state.

    Compute an initial state `zi` for the `lfilter` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    b, a : array_like (1-D)
        The IIR filter coefficients. See `lfilter` for more
        information.

    Returns
    -------
    zi : 1-D ndarray
        The initial state for the filter.

    See Also
    --------
    lfilter, lfiltic, filtfilt
    """
    a0 = a[0]
    a_r = - a[1:] / a0
    # b = b / a0
    num_b = b.size - 1
    num_a = a_r.size

    # The initial state for a FIR filter will be always one for a step input
    zi = cupy.ones(num_b)
    if num_a > 0:
        zi_t = cupy.r_[zi, cupy.zeros(num_a)]
        y, _ = lfilter(b, a, cupy.ones(num_a + 1), zi=zi_t)
        y1 = y[:num_a]
        y2 = y[-num_a:]

        C = compute_correction_factors(a_r, a_r.size + 1, a.dtype)
        C = C[:, a_r.size:]
        C1 = C[:, :a_r.size].T
        C2 = C[:, -a_r.size:].T

        # Take the difference between the non-adjusted output values and
        # compute which initial output state would cause them to be constant.
        y_zi = cupy.linalg.solve(C1 - C2, y2 - y1)
        zi = cupy.r_[zi, y_zi]
    return zi


def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    See Also
    --------
    scipy.signal.detrend


    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = cupy.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - cupy.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = cupy.sort(cupy.unique(cupy.r_[0, bp, N]))
        if cupy.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdims = tuple(cupy.r_[axis, 0:axis, axis + 1:rnk])
        newdata = data.transpose(newdims).reshape(N, -1)

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)

        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = cupy.ones((Npts, 2), dtype)
            A[:, 0] = cupy.arange(1, Npts + 1, dtype=dtype) / Npts
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - A @ coef

        # Put data back in original shape.
        tdshape = cupy.take(dshape, newdims, 0)
        ret = cupy.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = cupy.transpose(ret, tuple(olddims))
        return ret


def deconvolve(signal, divisor):
    """Deconvolves ``divisor`` out of ``signal`` using inverse filtering.

    Returns the quotient and remainder such that
    ``signal = convolve(divisor, quotient) + remainder``

    Parameters
    ----------
    signal : (N,) array_like
        Signal data, typically a recorded signal
    divisor : (N,) array_like
        Divisor data, typically an impulse response or filter that was
        applied to the original signal

    Returns
    -------
    quotient : ndarray
        Quotient, typically the recovered original signal
    remainder : ndarray
        Remainder

    See Also
    --------
    cupy.polydiv : performs polynomial division (same operation, but
                   also accepts poly1d objects)

    Examples
    --------
    Deconvolve a signal that's been filtered:

    >>> from cupyx.scipy import signal
    >>> original = [0, 1, 0, 0, 1, 1, 0, 0]
    >>> impulse_response = [2, 1]
    >>> recorded = signal.convolve(impulse_response, original)
    >>> recorded
    array([0, 2, 1, 0, 2, 3, 1, 0, 0])
    >>> recovered, remainder = signal.deconvolve(recorded, impulse_response)
    >>> recovered
    array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.])

    """
    num = cupy.atleast_1d(signal)
    den = cupy.atleast_1d(divisor)
    if num.ndim > 1:
        raise ValueError("signal must be 1-D.")
    if den.ndim > 1:
        raise ValueError("divisor must be 1-D.")
    N = len(num)
    D = len(den)
    if D > N:
        quot = []
        rem = num
    else:
        input = cupy.zeros(N - D + 1, float)
        input[0] = 1
        quot = lfilter(num, den, input)
        rem = num - convolve(den, quot, mode='full')
    return quot, rem


def _get_kernel_size(kernel_size, ndim):
    if kernel_size is None:
        kernel_size = (3,) * ndim
    kernel_size = _util._fix_sequence_arg(kernel_size, ndim,
                                          'kernel_size', int)
    if any((k % 2) != 1 for k in kernel_size):
        raise ValueError("Each element of kernel_size should be odd")
    return kernel_size
