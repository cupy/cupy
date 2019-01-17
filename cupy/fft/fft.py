import six

import numpy as np

import cupy
from cupy.cuda import cufft


def _convert_dtype(a, value_type):
    if value_type != 'R2C':
        if a.dtype in [np.float16, np.float32]:
            return a.astype(np.complex64)
        elif a.dtype not in [np.complex64, np.complex128]:
            return a.astype(np.complex128)
    else:
        if a.dtype in [np.complex64, np.complex128]:
            return a.real
        elif a.dtype == np.float16:
            return a.astype(np.float32)
        elif a.dtype not in [np.float32, np.float64]:
            return a.astype(np.float64)
    return a


def _cook_shape(a, s, axes, value_type):
    if s is None:
        return a
    if (value_type == 'C2R') and (s[-1] is not None):
        s = list(s)
        s[-1] = s[-1] // 2 + 1
    for sz, axis in zip(s, axes):
        if (sz is not None) and (sz != a.shape[axis]):
            shape = list(a.shape)
            if shape[axis] > sz:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, sz)
                a = a[index]
            else:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, shape[axis])
                shape[axis] = sz
                z = cupy.zeros(shape, a.dtype.char)
                z[index] = a
                a = z
    return a


def _convert_fft_type(a, value_type):
    if value_type == 'C2C' and a.dtype == np.complex64:
        return cufft.CUFFT_C2C
    elif value_type == 'R2C' and a.dtype == np.float32:
        return cufft.CUFFT_R2C
    elif value_type == 'C2R' and a.dtype == np.complex64:
        return cufft.CUFFT_C2R
    elif value_type == 'C2C' and a.dtype == np.complex128:
        return cufft.CUFFT_Z2Z
    elif value_type == 'R2C' and a.dtype == np.float64:
        return cufft.CUFFT_D2Z
    else:
        return cufft.CUFFT_Z2D


def _exec_fft(a, direction, value_type, norm, axis, out_size=None):
    fft_type = _convert_fft_type(a, value_type)

    if axis % a.ndim != a.ndim - 1:
        a = a.swapaxes(axis, -1)

    if a.base is not None or not a.flags.c_contiguous:
        a = a.copy()

    plan = cufft.Plan1d(a.shape[-1] if out_size is None else out_size,
                        fft_type, a.size // a.shape[-1])
    out = plan.get_output_array(a)
    plan.fft(a, out, direction)

    sz = out.shape[-1]
    if fft_type == cufft.CUFFT_R2C or fft_type == cufft.CUFFT_D2Z:
        sz = a.shape[-1]
    if norm is None:
        if direction == cufft.CUFFT_INVERSE:
            out /= sz
    else:
        out /= cupy.sqrt(sz)

    if axis % a.ndim != a.ndim - 1:
        out = out.swapaxes(axis, -1)

    return out


def _fft_c2c(a, direction, norm, axes):
    for axis in axes:
        a = _exec_fft(a, direction, 'C2C', norm, axis)
    return a


def _fft(a, s, axes, norm, direction, value_type='C2C'):
    if norm not in (None, 'ortho'):
        raise ValueError('Invalid norm value %s, should be None or \"ortho\".'
                         % norm)

    if s is not None:
        for n in s:
            if (n is not None) and (n < 1):
                raise ValueError(
                    "Invalid number of FFT data points (%d) specified." % n)

    if (s is not None) and (axes is not None) and len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")

    a = _convert_dtype(a, value_type)
    if axes is None:
        if s is None:
            dim = a.ndim
        else:
            dim = len(s)
        axes = [i for i in six.moves.range(-dim, 0)]
    a = _cook_shape(a, s, axes, value_type)

    if value_type == 'C2C':
        a = _fft_c2c(a, direction, norm, axes)
    elif value_type == 'R2C':
        a = _exec_fft(a, direction, value_type, norm, axes[-1])
        a = _fft_c2c(a, direction, norm, axes[:-1])
    else:
        a = _fft_c2c(a, direction, norm, axes[:-1])
        if (s is None) or (s[-1] is None):
            out_size = a.shape[axes[-1]] * 2 - 2
        else:
            out_size = s[-1]
        a = _exec_fft(a, direction, value_type, norm, axes[-1], out_size)

    return a


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
    return _fft(a, (n,), (axis,), norm, cupy.cuda.cufft.CUFFT_FORWARD)


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
    return _fft(a, (n,), (axis,), norm, cufft.CUFFT_INVERSE)


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
    return _fft(a, s, axes, norm, cufft.CUFFT_FORWARD)


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
    return _fft(a, s, axes, norm, cufft.CUFFT_INVERSE)


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
    return _fft(a, s, axes, norm, cufft.CUFFT_FORWARD)


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
    return _fft(a, s, axes, norm, cufft.CUFFT_INVERSE)


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
    return _fft(a, (n,), (axis,), norm, cufft.CUFFT_FORWARD, 'R2C')


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
    return _fft(a, (n,), (axis,), norm, cufft.CUFFT_INVERSE, 'C2R')


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
    return _fft(a, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C')


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
    return _fft(a, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R')


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
    return _fft(a, s, axes, norm, cufft.CUFFT_FORWARD, 'R2C')


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
    return _fft(a, s, axes, norm, cufft.CUFFT_INVERSE, 'C2R')


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
        x (cupy.ndarray): Input array.
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
        x (cupy.ndarray): Input array.
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
