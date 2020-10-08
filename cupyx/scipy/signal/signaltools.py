import warnings

import cupy

from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import filters
from cupyx.scipy.signal import _signaltools_core as _st_core


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
        raise ValueError("acceptable methods are 'auto', 'direct', or 'fft'")

    if method == 'auto':
        method = choose_conv_method(in1, in2, mode=mode)

    if method == 'direct':
        return _st_core._direct_correlate(in1, in2, mode, in1.dtype,
                                          convolution)

    # if method == 'fft':
    raise ValueError('fftconvolve currently not supported')


def choose_conv_method(in1, in2, mode='full'):
    """Find the fastest convolution/correlation method.

    Args:
        in1 (cupy.ndarray): first input.
        in2 (cupy.ndarray): second input.
        mode (str, optional): ``valid``, ``same``, ``full``.

    Returns:
        str: A string indicating which convolution method is fastest,
        either ``direct`` or ``fft1``.

    .. warning::
        This function currently doesn't support measure option,
        nor multidimensional inputs. It does not guarantee
        the compatibility of the return value to SciPy's one.

    .. seealso:: :func:`scipy.signal.choose_conv_method`

    """
    return cupy._math.misc._choose_conv_method(in1, in2, mode)


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
        raise ValueError("Acceptable boundary flags are 'fill' (or 'pad'), "
                         "'circular' (or 'wrap'), 'symmetric' (or 'symm'), "
                         "and 'reflect'.")
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
    if im.dtype.kind == 'c':
        # TODO: adding support for complex types requires ndimage filters
        # to support complex types (which they could easily if not for the
        # scipy compatibility requirement of forbidding complex and using
        # float64 intermediates)
        raise TypeError("complex types not currently supported")
    if mysize is None:
        mysize = 3
    mysize = _util._fix_sequence_arg(mysize, im.ndim, 'mysize', int)
    im = im.astype(float, copy=False)

    # Estimate the local mean
    local_mean = filters.uniform_filter(im, mysize, mode='constant')

    # Estimate the local variance
    local_var = filters.uniform_filter(im*im, mysize, mode='constant')
    local_var -= local_mean*local_mean

    # Estimate the noise power if needed.
    if noise is None:
        noise = local_var.mean()

    # Perform the filtering
    res = im - local_mean
    res *= (1 - noise / local_var)
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
    return filters.rank_filter(a, rank, footprint=domain, mode='constant')


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
        # (and filters.rank_filter raise TypeError)
        raise ValueError("complex types not supported")
    # output is forced to float64 to match scipy
    kernel_size = _get_kernel_size(kernel_size, volume.ndim)
    if any(k > s for k, s in zip(kernel_size, volume.shape)):
        warnings.warn('kernel_size exceeds volume extent: '
                      'volume will be zero-padded')

    size = cupy.core.internal.prod(kernel_size)
    return filters.rank_filter(volume, size // 2, size=kernel_size,
                               output=float, mode='constant')


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
    if input.dtype not in (cupy.uint8, cupy.float32, cupy.float64):
        # Scipy's version only supports uint8, float32, and float64
        raise ValueError("only supports uint8, float32, and float64")
    if input.ndim != 2:
        raise ValueError('input must be 2d')
    kernel_size = _get_kernel_size(kernel_size, input.ndim)
    order = kernel_size[0] * kernel_size[1] // 2
    return filters.rank_filter(input, order, size=kernel_size, mode='constant')


def _get_kernel_size(kernel_size, ndim):
    if kernel_size is None:
        kernel_size = (3,) * ndim
    kernel_size = _util._fix_sequence_arg(kernel_size, ndim,
                                          'kernel_size', int)
    if any((k % 2) != 1 for k in kernel_size):
        raise ValueError("Each element of kernel_size should be odd")
    return kernel_size
