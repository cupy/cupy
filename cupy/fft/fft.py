import six

import numpy as np

import cupy
from cupy import cufft


def fft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.fft`
    """
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional inverse FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. If ``n``
            is not given, the length of the input along the axis specified by
            ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.ifft`
    """
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_INVERSE)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.fft2`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional inverse FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.ifft2`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE)


def fftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.fftn`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD)


def ifftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional inverse FFT.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the transformed axes of the
            output. If ``s`` is not given, the lengths of the input along the
            axes specified by ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other.

    .. seealso:: :func:`numpy.fft.ifftn`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE)


def rfft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Number of points along transformation axis in the
            input to use. If ``n`` is not given, the length of the input along
            the axis specified by ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. The length of the
            transformed axis is ``n//2+1``.

    .. seealso:: :func:`numpy.fft.rfft`
    """
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_FORWARD,
                     'R2C')


def irfft(a, n=None, axis=-1, norm=None):
    """Compute the one-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. For
            ``n`` output points, ``n//2+1`` input points are necessary. If
            ``n`` is not given, it is determined from the length of the input
            along the axis specified by ``axis``.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. If ``n`` is not
            given, the length of the transformed axis is`2*(m-1)` where `m`
            is the length of the transformed axis of the input.

    .. seealso:: :func:`numpy.fft.irfft`
    """
    return cufft.fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_INVERSE,
                     'C2R')


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`numpy.fft.rfft2`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD, 'R2C')


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the two-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            `2*(m-1)` where `m` is the length of the final transformed axis of
            the input.

    .. seealso:: :func:`numpy.fft.irfft2`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE, 'C2R')


def rfftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape to use from the input. If ``s`` is not
            given, the lengths of the input along the axes specified by
            ``axes`` are used.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. The length of the
            last axis transformed will be ``s[-1]//2+1``.

    .. seealso:: :func:`numpy.fft.rfftn`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_FORWARD, 'R2C')


def irfftn(a, s=None, axes=None, norm=None):
    """Compute the N-dimensional inverse FFT for real input.

    Args:
        a (cupy.ndarray): Array to be transform.
        s (None or tuple of ints): Shape of the output. If ``s`` is not given,
            they are determined from the lengths of the input along the axes
            specified by ``axes``.
        axes (tuple of ints): Axes over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``s`` and type
            will convert to complex if the input is other. If ``s`` is not
            given, the length of final transformed axis of output will be
            ``2*(m-1)`` where `m` is the length of the final transformed axis
            of the input.

    .. seealso:: :func:`numpy.fft.irfftn`
    """
    return cufft.fft(a, s, axes, norm, cupy.cuda.cufft.CUFFT_INVERSE, 'C2R')


def hfft(a, n=None, axis=-1, norm=None):
    """Compute the FFT of a signal that has Hermitian symmetry.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Length of the transformed axis of the output. For
            ``n`` output points, ``n//2+1`` input points are necessary. If
            ``n`` is not given, it is determined from the length of the input
            along the axis specified by ``axis``.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. If ``n`` is not
            given, the length of the transformed axis is ``2*(m-1)`` where `m`
            is the length of the transformed axis of the input.

    .. seealso:: :func:`numpy.fft.hfft`
    """
    a = irfft(a.conj(), n, axis)
    return a * (a.shape[axis] if norm is None else
                cupy.sqrt(a.shape[axis], dtype=a.dtype))


def ihfft(a, n=None, axis=-1, norm=None):
    """Compute the FFT of a signal that has Hermitian symmetry.

    Args:
        a (cupy.ndarray): Array to be transform.
        n (None or int): Number of points along transformation axis in the
            input to use. If ``n`` is not given, the length of the input along
            the axis specified by ``axis`` is used.
        axis (int): Axis over which to compute the FFT.
        norm (None or ``"ortho"``): Keyword to specify the normalization mode.

    Returns:
        cupy.ndarray:
            The transformed array which shape is specified by ``n`` and type
            will convert to complex if the input is other. The length of the
            transformed axis is ``n//2+1``.

    .. seealso:: :func:`numpy.fft.ihfft`
    """
    if n is None:
        n = a.shape[axis]
    return rfft(a, n, axis, norm).conj() / (n if norm is None else 1)


def fftfreq(n, d=1.0):
    """Return the FFT sample frequencies.

    Args:
        n (int): Window length.
        d (scalar): Sample spacing.

    Returns:
        cupy.ndarray: Array of length ``n`` containing the sample frequencies.

    .. seealso:: :func:`numpy.fft.fftfreq`
    """
    return cupy.hstack((cupy.arange(0, (n - 1) // 2 + 1, dtype=np.float64),
                        cupy.arange(-(n // 2), 0, dtype=np.float64))) / n / d


def rfftfreq(n, d=1.0):
    """Return the FFT sample frequencies for real input.

    Args:
        n (int): Window length.
        d (scalar): Sample spacing.

    Returns:
        cupy.ndarray:
            Array of length ``n//2+1`` containing the sample frequencies.

    .. seealso:: :func:`numpy.fft.rfftfreq`
    """
    return cupy.arange(0, n // 2 + 1, dtype=np.float64) / n / d


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum.

    Args:
        x (cumpy.adarray): Input array.
        axes (int or tuple of ints): Axes over which to shift. Default is
            ``None``, which shifts all axes.

    Returns:
        cupy.ndarray: The shifted array.

    .. seealso:: :func:`numpy.fft.fftshift`
    """
    x = cupy.asarray(x)
    if axes is None:
        axes = list(six.moves.range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, x.shape[axis] // 2, axis)
    return x


def ifftshift(x, axes=None):
    """The inverse of :meth:`fftshift`.

    Args:
        x (cumpy.adarray): Input array.
        axes (int or tuple of ints): Axes over which to shift. Default is
            ``None``, which shifts all axes.

    Returns:
        cupy.ndarray: The shifted array.

    .. seealso:: :func:`numpy.fft.ifftshift`
    """
    x = cupy.asarray(x)
    if axes is None:
        axes = list(six.moves.range(x.ndim))
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)
    for axis in axes:
        x = cupy.roll(x, -(x.shape[axis] // 2), axis)
    return x
