import numpy

import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy import special


def _get_output_fourier(output, input, complex_only=False):
    types = [cupy.complex64, cupy.complex128]
    if not complex_only:
        types += [cupy.float32, cupy.float64]

    if output is None:
        if input.dtype in types:
            output = cupy.zeros(input.shape, dtype=input.dtype)
        else:
            output = cupy.zeros(input.shape, dtype=types[-1])
    elif type(output) is type:
        if output not in types:
            raise RuntimeError('output type not supported')
        output = cupy.zeros(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError('output shape not correct')
    return output


def _reshape_nd(arr, ndim, axis):
    """Promote a 1d array to ndim with non-singleton size along axis."""
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)


def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    """Multidimensional Gaussian shift filter.

    The array is multiplied with the Fourier transform of a (separable)
    Gaussian kernel.

    Args:
        input (cupy.ndarray): The input array.
        sigma (float or sequence of float):  The sigma of the Gaussian kernel.
            If a float, `sigma` is the same for all axes. If a sequence,
            `sigma` has to contain one value for each axis.
        n (int, optional):  If `n` is negative (default), then the input is
            assumed to be the result of a complex fft. If `n` is larger than or
            equal to zero, the input is assumed to be the result of a real fft,
            and `n` gives the length of the array before transformation along
            the real transform direction.
        axis (int, optional): The axis of the real transform (only used when
            ``n > -1``).
        output (cupy.ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (cupy.ndarray): The filtered output.
    """
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sigmas = _util._fix_sequence_arg(sigma, ndim, 'sigma')

    output[...] = input
    for ax, (sigmak, ax_size) in enumerate(zip(sigmas, output.shape)):

        # compute the frequency grid in Hz
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = cupy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)

        # compute the Gaussian weights
        arr *= arr
        scale = sigmak * sigmak / -2
        arr *= (4 * numpy.pi * numpy.pi) * scale
        cupy.exp(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output


def fourier_uniform(input, size, n=-1, axis=-1, output=None):
    """Multidimensional uniform shift filter.

    The array is multiplied with the Fourier transform of a box of given size.

    Args:
        input (cupy.ndarray): The input array.
        size (float or sequence of float):  The sigma of the box used for
            filtering. If a float, `size` is the same for all axes. If a
            sequence, `size` has to contain one value for each axis.
        n (int, optional):  If `n` is negative (default), then the input is
            assumed to be the result of a complex fft. If `n` is larger than or
            equal to zero, the input is assumed to be the result of a real fft,
            and `n` gives the length of the array before transformation along
            the real transform direction.
        axis (int, optional): The axis of the real transform (only used when
            ``n > -1``).
        output (cupy.ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (cupy.ndarray): The filtered output.
    """
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sizes = _util._fix_sequence_arg(size, ndim, 'size')

    output[...] = input
    for ax, (size, ax_size) in enumerate(zip(sizes, output.shape)):

        # compute the frequency grid in Hz
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = cupy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)

        # compute the uniform filter weights
        arr *= size
        cupy.sinc(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output


def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    """Multidimensional Fourier shift filter.

    The array is multiplied with the Fourier transform of a shift operation.

    Args:
        input (cupy.ndarray): The input array. This should be in the Fourier
            domain.
        shift (float or sequence of float):  The size of shift. If a float,
            `shift` is the same for all axes. If a sequence, `shift` has to
            contain one value for each axis.
        n (int, optional):  If `n` is negative (default), then the input is
            assumed to be the result of a complex fft. If `n` is larger than or
            equal to zero, the input is assumed to be the result of a real fft,
            and `n` gives the length of the array before transformation along
            the real transform direction.
        axis (int, optional): The axis of the real transform (only used when
            ``n > -1``).
        output (cupy.ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (cupy.ndarray): The shifted output (in the Fourier domain).
    """
    ndim = input.ndim
    output = _get_output_fourier(output, input, complex_only=True)
    axis = internal._normalize_axis_index(axis, ndim)
    shifts = _util._fix_sequence_arg(shift, ndim, 'shift')

    output[...] = input
    for ax, (shiftk, ax_size) in enumerate(zip(shifts, output.shape)):
        if shiftk == 0:
            continue
        if ax == axis and n > 0:
            # cp.fft.rfftfreq(ax_size) * (-2j * numpy.pi * shiftk *  ax_size/n)
            arr = cupy.arange(ax_size, dtype=output.dtype)
            arr *= -2j * numpy.pi * shiftk / n
        else:
            arr = cupy.fft.fftfreq(ax_size)
            arr = arr * (-2j * numpy.pi * shiftk)
        cupy.exp(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output


def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
    """Multidimensional ellipsoid Fourier filter.

    The array is multiplied with the fourier transform of a ellipsoid of
    given sizes.

    Args:
        input (cupy.ndarray): The input array.
        size (float or sequence of float):  The size of the box used for
            filtering. If a float, `size` is the same for all axes. If a
            sequence, `size` has to contain one value for each axis.
        n (int, optional):  If `n` is negative (default), then the input is
            assumed to be the result of a complex fft. If `n` is larger than or
            equal to zero, the input is assumed to be the result of a real fft,
            and `n` gives the length of the array before transformation along
            the real transform direction.
        axis (int, optional): The axis of the real transform (only used when
            ``n > -1``).
        output (cupy.ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (cupy.ndarray): The filtered output.
    """
    ndim = input.ndim
    if ndim == 1:
        return fourier_uniform(input, size, n, axis, output)

    if ndim > 3:
        # Note: SciPy currently does not do any filtering on >=4d inputs, but
        #       does not warn about this!
        raise NotImplementedError('Only 1d, 2d and 3d inputs are supported')
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sizes = _util._fix_sequence_arg(size, ndim, 'size')

    output[...] = input

    # compute the distance from the origin for all samples in Fourier space
    distance = 0
    for ax, (size, ax_size) in enumerate(zip(sizes, output.shape)):
        # compute the frequency grid in Hz
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr *= numpy.pi * size / n
        else:
            arr = cupy.fft.fftfreq(ax_size)
            arr *= numpy.pi * size
        arr = arr.astype(output.real.dtype, copy=False)
        arr *= arr
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        distance = distance + arr
    cupy.sqrt(distance, out=distance)

    if ndim == 2:
        special.j1(distance, out=output)
        output *= 2
        output /= distance
    elif ndim == 3:
        cupy.sin(distance, out=output)
        output -= distance * cupy.cos(distance)
        output *= 3
        output /= distance ** 3
    output[(0,) * ndim] = 1.0  # avoid NaN in corner at frequency=0 location
    output *= input

    return output
