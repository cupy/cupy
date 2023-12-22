import warnings

import cupy
from cupy._core import internal
from cupy.linalg import lstsq

import cupyx.scipy.fft as sp_fft
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
from cupyx.scipy.signal._arraytools import (
    const_ext, even_ext, odd_ext, axis_reverse, axis_slice, axis_assign)
from cupyx.scipy.signal._iir_utils import (
    apply_iir, apply_iir_sos, compute_correction_factors,
    compute_correction_factors_sos)


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


def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.

    Parameters
    ----------
    in1_len : int
        First input size.
    in2_len : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.

    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.

    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    scipy.signal.correlation_lags
    """
    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = cupy.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = cupy.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid - lag_bound):(mid + lag_bound)]
        else:
            lags = lags[(mid - lag_bound):(mid + lag_bound) + 1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = cupy.arange(lag_bound + 1)
        else:
            lags = cupy.arange(lag_bound, 1)
    return lags


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
    if volume.dtype.char == 'e':
        # scipy doesn't support float16
        raise ValueError("float16 type not supported")
    if volume.dtype.kind == 'b':
        # scipy doesn't support bool
        raise ValueError("bool type not supported")
    kernel_size = _get_kernel_size(kernel_size, volume.ndim)
    if volume.dtype == 'F':
        raise TypeError("complex types not supported")
    if volume.dtype.kind == 'c':
        # scipy doesn't support complex
        raise ValueError("complex types not supported")
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
    if input.dtype.char == 'e':
        # scipy doesn't support float16
        raise ValueError("float16 type not supported")
    if input.dtype.kind == 'b':
        # scipy doesn't support bool
        raise ValueError("bool type not supported")
    if input.ndim != 2:
        raise ValueError('input must be 2d')
    kernel_size = _get_kernel_size(kernel_size, input.ndim)
    if input.dtype == 'F':
        raise TypeError("complex types not supported")
    if input.dtype.kind == 'c':
        # scipy doesn't support complex
        raise ValueError("complex types not supported")
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

    x_full = cupy.zeros(pad_shape, dtype=fir_dtype)
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
        zero_coef = cupy.where(a_r == 0)[0]

        C = compute_correction_factors(a_r, a_r.size + 1, a_r.dtype)
        C = C[:, a_r.size:]
        C1 = C[:, :a_r.size].T
        C2 = C[:, -a_r.size:].T

        # Take the difference between the non-adjusted output values and
        # compute which initial output state would cause them to be constant.
        if not len(zero_coef):
            y_zi = cupy.linalg.solve(C1 - C2, y2 - y1)
        else:
            # Any zero coefficient would cause the system to be underdetermined
            # therefore a least square solution is computed instead.
            y_zi, _, _, _ = cupy.linalg.lstsq(C1 - C2, y2 - y1, rcond=None)

        y_zi = cupy.nan_to_num(y_zi, nan=0, posinf=cupy.inf, neginf=-cupy.inf)
        zi = cupy.r_[zi, y_zi[::-1]]
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
        bp = bp.tolist()
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = cupy.moveaxis(data, axis, 0)
        newdata_shape = newdata.shape
        newdata = newdata.reshape(N, -1)

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)

        # Find leastsq fit and remove it for each piece
        for m in range(len(bp) - 1):
            Npts = bp[m + 1] - bp[m]
            A = cupy.ones((Npts, 2), dtype)
            A[:, 0] = cupy.arange(1, Npts + 1, dtype=dtype) / Npts
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = lstsq(A, newdata[sl], rcond=None)
            newdata[sl] = newdata[sl] - A @ coef

        # Put data back in original shape.
        newdata = newdata.reshape(newdata_shape)
        ret = cupy.moveaxis(newdata, 0, axis)
        return ret


def _filtfilt_gust(b, a, x, axis=-1, irlen=None):
    """Forward-backward IIR filter that uses Gustafsson's method.

    Apply the IIR filter defined by `(b,a)` to `x` twice, first forward
    then backward, using Gustafsson's initial conditions [1]_.

    Let ``y_fb`` be the result of filtering first forward and then backward,
    and let ``y_bf`` be the result of filtering first backward then forward.
    Gustafsson's method is to compute initial conditions for the forward
    pass and the backward pass such that ``y_fb == y_bf``.

    Parameters
    ----------
    b : scalar or 1-D ndarray
        Numerator coefficients of the filter.
    a : scalar or 1-D ndarray
        Denominator coefficients of the filter.
    x : ndarray
        Data to be filtered.
    axis : int, optional
        Axis of `x` to be filtered.  Default is -1.
    irlen : int or None, optional
        The length of the nonnegligible part of the impulse response.
        If `irlen` is None, or if the length of the signal is less than
        ``2 * irlen``, then no part of the impulse response is ignored.

    Returns
    -------
    y : ndarray
        The filtered data.
    x0 : ndarray
        Initial condition for the forward filter.
    x1 : ndarray
        Initial condition for the backward filter.

    Notes
    -----
    Typically the return values `x0` and `x1` are not needed by the
    caller.  The intended use of these return values is in unit tests.

    References
    ----------
    .. [1] F. Gustaffson. Determining the initial states in forward-backward
           filtering. Transactions on Signal Processing, 46(4):988-992, 1996.
    """
    # In the comments, "Gustafsson's paper" and [1] refer to the
    # paper referenced in the docstring.

    b = cupy.atleast_1d(b)
    a = cupy.atleast_1d(a)

    order = max(len(b), len(a)) - 1
    if order == 0:
        # The filter is just scalar multiplication, with no state.
        scale = (b[0] / a[0]) ** 2
        y = scale * x
        return y, cupy.array([]), cupy.array([])

    if axis != -1 or axis != x.ndim - 1:
        # Move the axis containing the data to the end.
        x = cupy.swapaxes(x, axis, x.ndim - 1)

    # n is the number of samples in the data to be filtered.
    n = x.shape[-1]

    if irlen is None or n <= 2 * irlen:
        m = n
    else:
        m = irlen

    # Create Obs, the observability matrix (called O in the paper).
    # This matrix can be interpreted as the operator that propagates
    # an arbitrary initial state to the output, assuming the input is
    # zero.
    # In Gustafsson's paper, the forward and backward filters are not
    # necessarily the same, so he has both O_f and O_b.  We use the same
    # filter in both directions, so we only need O. The same comment
    # applies to S below.
    Obs = cupy.zeros((m, order))
    x_in = cupy.zeros(m)
    x_in[0] = 1
    Obs[:, 0] = lfilter(cupy.ones(1), a, x_in)
    for k in range(1, order):
        Obs[k:, k] = Obs[:-k, 0]

    # Obsr is O^R (Gustafsson's notation for row-reversed O)
    Obsr = Obs[::-1]

    # Create S.  S is the matrix that applies the filter to the reversed
    # propagated initial conditions.  That is,
    #     out = S.dot(zi)
    # is the same as
    #     tmp, _ = lfilter(b, a, zeros(), zi=zi)  # Propagate ICs.
    #     out = lfilter(b, a, tmp[::-1])          # Reverse and filter.

    # Equations (5) & (6) of [1]
    S = lfilter(b, a, Obs[::-1], axis=0)

    # Sr is S^R (row-reversed S)
    Sr = S[::-1]

    # M is [(S^R - O), (O^R - S)]
    if m == n:
        M = cupy.hstack((Sr - Obs, Obsr - S))
    else:
        # Matrix described in section IV of [1].
        M = cupy.zeros((2*m, 2*order))
        M[:m, :order] = Sr - Obs
        M[m:, order:] = Obsr - S

    # Naive forward-backward and backward-forward filters.
    # These have large transients because the filters use zero initial
    # conditions.
    y_f = lfilter(b, a, x)
    y_fb = lfilter(b, a, y_f[..., ::-1])[..., ::-1]

    y_b = lfilter(b, a, x[..., ::-1])[..., ::-1]
    y_bf = lfilter(b, a, y_b)

    delta_y_bf_fb = y_bf - y_fb
    if m == n:
        delta = delta_y_bf_fb
    else:
        start_m = delta_y_bf_fb[..., :m]
        end_m = delta_y_bf_fb[..., -m:]
        delta = cupy.concatenate((start_m, end_m), axis=-1)

    # ic_opt holds the "optimal" initial conditions.
    # The following code computes the result shown in the formula
    # of the paper between equations (6) and (7).
    if delta.ndim == 1:
        ic_opt = cupy.linalg.lstsq(M, delta, rcond=None)[0]
    else:
        # Reshape delta so it can be used as an array of multiple
        # right-hand-sides in linalg.lstsq.
        delta2d = delta.reshape(-1, delta.shape[-1]).T
        ic_opt0 = cupy.linalg.lstsq(M, delta2d, rcond=None)[0].T
        ic_opt = ic_opt0.reshape(delta.shape[:-1] + (M.shape[-1],))

    # Now compute the filtered signal using equation (7) of [1].
    # First, form [S^R, O^R] and call it W.
    if m == n:
        W = cupy.hstack((Sr, Obsr))
    else:
        W = cupy.zeros((2*m, 2*order))
        W[:m, :order] = Sr
        W[m:, order:] = Obsr

    # Equation (7) of [1] says
    #     Y_fb^opt = Y_fb^0 + W * [x_0^opt; x_{N-1}^opt]
    # `wic` is (almost) the product on the right.
    # W has shape (m, 2*order), and ic_opt has shape (..., 2*order),
    # so we can't use W.dot(ic_opt).  Instead, we dot ic_opt with W.T,
    # so wic has shape (..., m).
    wic = ic_opt.dot(W.T)

    # `wic` is "almost" the product of W and the optimal ICs in equation
    # (7)--if we're using a truncated impulse response (m < n), `wic`
    # contains only the adjustments required for the ends of the signal.
    # Here we form y_opt, taking this into account if necessary.
    y_opt = y_fb
    if m == n:
        y_opt += wic
    else:
        y_opt[..., :m] += wic[..., :m]
        y_opt[..., -m:] += wic[..., -m:]

    x0 = ic_opt[..., :order]
    x1 = ic_opt[..., -order:]
    if axis != -1 or axis != x.ndim - 1:
        # Restore the data axis to its original position.
        x0 = cupy.swapaxes(x0, axis, x.ndim - 1)
        x1 = cupy.swapaxes(x1, axis, x.ndim - 1)
        y_opt = cupy.swapaxes(y_opt, axis, x.ndim - 1)

    return y_opt, x0, x1


def _validate_pad(padtype, padlen, x, axis, ntaps):
    """Helper to validate padding for filtfilt"""
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype "
                          "must be 'even', 'odd', 'constant', or None.") %
                         padtype)

    if padtype is None:
        padlen = 0

    if padlen is None:
        # Original padding; preserved for backwards compatibility.
        edge = ntaps * 3
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[axis] <= edge:
        raise ValueError("The length of the input vector x must be greater "
                         "than padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=axis)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=axis)
        else:
            ext = const_ext(x, edge, axis=axis)
    else:
        ext = x
    return edge, ext


def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad',
             irlen=None):
    """
    Apply a digital filter forward and backward to a signal.

    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.

    The function provides options for handling the edges of the signal.

    The function `sosfiltfilt` (and filter design using ``output='sos'``)
    should be preferred over `filtfilt` for most filtering tasks, as
    second-order sections have fewer numerical problems.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.

    See Also
    --------
    sosfiltfilt, lfilter_zi, lfilter, lfiltic, savgol_filter, sosfilt

    Notes
    -----
    When `method` is "pad", the function pads the data along the given axis
    in one of three ways: odd, even or constant.  The odd and even extensions
    have the corresponding symmetry about the end point of the data.  The
    constant extension extends the data with the values at the end points. On
    both the forward and backward passes, the initial condition of the
    filter is found by using `lfilter_zi` and scaling it by the end point of
    the extended data.

    When `method` is "gust", Gustafsson's method [1]_ is used.  Initial
    conditions are chosen for the forward and backward passes so that the
    forward-backward filter gives the same result as the backward-forward
    filter.

    References
    ----------
    .. [1] F. Gustaffson, "Determining the initial states in forward-backward
           filtering", Transactions on Signal Processing, Vol. 46, pp. 988-992,
           1996.
    """
    b = cupy.atleast_1d(b)
    a = cupy.atleast_1d(a)
    x = cupy.asarray(x)

    if method not in {"pad", "gust"}:
        raise ValueError("method must be 'pad' or 'gust'.")

    const_dtype = cupy.dtype(a.dtype)
    if const_dtype.kind == 'u':
        const_dtype = cupy.dtype(const_dtype.char.lower())
        a = a.astype(const_dtype)

    if method == "gust":
        y, z1, z2 = _filtfilt_gust(b, a, x, axis=axis, irlen=irlen)
        return y

    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=max(len(a), len(b)))

    # Get the steady state of the filter's step response.
    zi = lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = cupy.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y


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


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = cupy.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (cupy.abs(sos[:, 3] - 1.0) <= 1e-15).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


def _validate_x(x):
    x = cupy.asarray(x)
    if x.ndim == 0:
        raise ValueError('x must be at least 1-D')
    return x


def sosfilt(sos, x, axis=-1, zi=None):
    """
    Filter data along one dimension using cascaded second-order sections.

    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 4, ...)``, where
        ``..., 4, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 4.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz
    """
    x_ndim = x.ndim
    axis = internal._normalize_axis_index(axis, x_ndim)
    out = x

    out = apply_iir_sos(out, sos, axis, zi)
    return out


def sosfilt_zi(sos):
    """
    Construct initial conditions for sosfilt for step response steady-state.

    Compute an initial state `zi` for the `sosfilt` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    zi : ndarray
        Initial conditions suitable for use with ``sosfilt``, shape
        ``(n_sections, 4)``.

    See Also
    --------
    sosfilt, zpk2sos
    """
    n_sections = sos.shape[0]

    C = compute_correction_factors_sos(sos, 3, sos.dtype)
    zi = cupy.zeros((sos.shape[0], 4), dtype=sos.dtype)

    # The initial state for a FIR filter will be always one for a step input
    x_s = cupy.ones(3, dtype=sos.dtype)
    for s in range(n_sections):
        zi_s = cupy.atleast_2d(zi[s])
        sos_s = cupy.atleast_2d(sos[s])

        # The FIR starting value that guarantees a constant output will be
        # the same constant input values.
        zi_s[0, :2] = x_s[:2]

        # Find the non-adjusted values after applying the IIR filter.
        y_s, _ = sosfilt(sos_s, x_s, zi=zi_s)

        C_s = C[s]
        y1 = y_s[:2]
        y2 = y_s[-2:]
        C1 = C_s[:, :2].T
        C2 = C_s[:, -2:].T

        zero_iir_coef = cupy.where(sos[s, 3:] == 0)[0]

        # Take the difference between the non-adjusted output values and
        # compute which initial output state would cause them to be constant.
        if not len(zero_iir_coef):
            y_zi = cupy.linalg.solve(C1 - C2, y2 - y1)
        else:
            # Any zero coefficient would cause the system to be underdetermined
            # therefore a least square solution is computed instead.
            y_zi, _, _, _ = cupy.linalg.lstsq(C1 - C2, y2 - y1, rcond=None)

        y_zi = cupy.nan_to_num(y_zi, nan=0, posinf=cupy.inf, neginf=-cupy.inf)
        zi_s[0, 2:] = y_zi[::-1]
        x_s, _ = sosfilt(sos_s, x_s, zi=zi_s)

    return zi


def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    """
    A forward-backward digital filter using cascaded second-order sections.

    See `filtfilt` for more complete information about this method.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        The array of data to be filtered.
    axis : int, optional
        The axis of `x` to which the filter is applied.
        Default is -1.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `x` at both ends of
        `axis` before applying the filter.  This value must be less than
        ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding.
        The default value is::

            3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                        (sos[:, 5] == 0).sum()))

        The extra subtraction at the end attempts to compensate for poles
        and zeros at the origin (e.g. for odd-order filters) to yield
        equivalent estimates of `padlen` to those of `filtfilt` for
        second-order section filters built with `scipy.signal` functions.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `x`.

    See Also
    --------
    filtfilt, sosfilt, sosfilt_zi, sosfreqz
    """
    sos, n_sections = _validate_sos(sos)
    x = _validate_x(x)

    # `method` is "pad"...
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum().item(), (sos[:, 5] == 0).sum().item())
    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=ntaps)

    # These steps follow the same form as filtfilt with modifications
    zi = sosfilt_zi(sos)  # shape (n_sections, 4) --> (n_sections, ..., 4, ...)
    zi_shape = [1] * x.ndim
    zi_shape[axis] = 4
    zi.shape = [n_sections] + zi_shape
    x_0 = axis_slice(ext, stop=1, axis=axis)
    (y, zf) = sosfilt(sos, ext, axis=axis, zi=zi * x_0)
    y_0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = sosfilt(sos, axis_reverse(y, axis=axis), axis=axis, zi=zi * y_0)
    y = axis_reverse(y, axis=axis)
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)
    return y


def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : ndarray
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal

    See Also
    --------
    scipy.signal.hilbert

    """
    if cupy.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = sp_fft.fft(x, N, axis=axis)
    h = cupy.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [cupy.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = sp_fft.ifft(Xf * h, axis=axis)
    return x


def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : ndarray
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    See Also
    --------
    scipy.signal.hilbert2

    """
    if x.ndim < 2:
        x = cupy.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
    if cupy.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or (N[0] <= 0 or N[1] <= 0):
        raise ValueError("When given as a tuple, N must hold exactly "
                         "two positive integers")

    Xf = sp_fft.fft2(x, N, axes=(0, 1))
    h1 = cupy.zeros(N[0], dtype=Xf.dtype)
    h2 = cupy.zeros(N[1], dtype=Xf.dtype)
    for h in (h1, h1):
        N1 = h.shape[0]
        if N1 % 2 == 0:
            h[0] = h[N1 // 2] = 1
            h[1:N1 // 2] = 2
        else:
            h[0] = 1
            h[1:(N1 + 1) // 2] = 2

    h = h1[:, cupy.newaxis] * h2[cupy.newaxis, :]
    k = x.ndim
    while k > 2:
        h = h[:, cupy.newaxis]
        k -= 1
    x = sp_fft.ifft2(Xf * h, axes=(0, 1))
    return x
