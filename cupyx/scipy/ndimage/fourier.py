import cupy
import numpy as np


def _get_output_fourier_complex(output, input):
    if output is None:
        if input.dtype.type in [cupy.complex64, cupy.complex128]:
            output = cupy.zeros(input.shape, dtype=input.dtype)
        else:
            output = cupy.zeros(input.shape, dtype=cupy.complex128)
    elif type(output) is type:
        if output not in [cupy.complex64, cupy.complex128]:
            raise RuntimeError("output type not supported")
        output = cupy.zeros(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError("output shape not correct")
    return output


def _reshape_nd(arr, ndim, axis):
    """Promote a 1d array to ndim with non-singleton size along axis."""
    if arr.ndim != 1:
        raise ValueError("expected a 1d array")
    if axis < -ndim or axis > ndim - 1:
        raise ValueError("invalid axis")
    if ndim < 1:
        raise ValueError("ndim must be >= 1")
    axis = axis % ndim
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)


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
        output (ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (ndarray or None): The shifted output (in the Fourier domain).

    Returns
    -------
    fourier_shift : ndarray
        The shifted input.

    """
    output = _get_output_fourier_complex(output, input)
    output[...] = input

    ndim = output.ndim
    if axis < -ndim or axis >= ndim:
        raise ValueError("invalid axis")
    axis = axis % ndim

    if np.isscalar(shift):
        shift = (shift,) * ndim
    elif len(shift) != ndim:
        raise ValueError("number of shifts must match input.ndim")

    for kk in range(ndim):
        ax_size = output.shape[kk]
        shiftk = shift[kk]
        if shiftk == 0:
            continue
        if kk == axis and n > 0:
            # cp.fft.rfftfreq(ax_size) * (-2j * np.pi * shiftk *  ax_size / n)
            arr = cupy.arange(ax_size, dtype=output.dtype)
            arr *= -2j * np.pi * shiftk / n
        else:
            arr = cupy.fft.fftfreq(ax_size)
            arr = arr * (-2j * np.pi * shiftk)
        cupy.exp(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=kk)
        output *= arr

    return output
