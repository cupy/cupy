import numpy as np

import cupy
from cupyx.scipy.ndimage import filters


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
        output (ndarray): The shifted output (in the Fourier domain).
    """
    ndim = input.ndim
    output = _get_output_fourier_complex(output, input)
    axis = cupy.util._normalize_axis_index(axis, ndim)
    shifts = filters._fix_sequence_arg(shift, ndim, 'shift')

    output[...] = input
    for ax, (shiftk, ax_size) in enumerate(zip(shifts, output.shape)):
        if shiftk == 0:
            continue
        if ax == axis and n > 0:
            # cp.fft.rfftfreq(ax_size) * (-2j * np.pi * shiftk *  ax_size / n)
            arr = cupy.arange(ax_size, dtype=output.dtype)
            arr *= -2j * np.pi * shiftk / n
        else:
            arr = cupy.fft.fftfreq(ax_size)
            arr = arr * (-2j * np.pi * shiftk)
        cupy.exp(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output
