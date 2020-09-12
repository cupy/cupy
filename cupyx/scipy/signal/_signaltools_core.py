import cupy
from cupyx.scipy.ndimage import filters
from cupyx.scipy.ndimage import _util


def _check_conv_inputs(in1, in2, mode, convolution=True):
    if in1.ndim == in2.ndim == 0:
        return in1 * (in2 if convolution else in2.conj())
    if in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    if in1.size == 0 or in2.size == 0:
        return cupy.array([], dtype=in1.dtype)
    if mode not in ('full', 'same', 'valid'):
        raise ValueError("acceptable modes are 'valid', 'same', or 'full'")
    return None


def _direct_correlate(in1, in2, mode='full', output=float, convolution=False,
                      boundary='constant', fillvalue=0.0, shift=False):
    if in1.ndim != 1 and (in1.dtype.kind == 'b' or
                          (in1.dtype.kind == 'f' and in1.dtype.itemsize < 4)):
        raise ValueError('unsupported type in SciPy')

    # Swaps inputs so smaller one is in2:
    # NOTE: when mode != 'valid' we can only swap with a constant-0 boundary
    swapped_inputs = False
    orig_in1_shape = in1.shape
    if _inputs_swap_needed(mode, in1.shape, in2.shape) or (
            in2.size > in1.size and boundary == 'constant' and fillvalue == 0):
        in1, in2 = in2, in1
        swapped_inputs = not convolution

    # Due to several optimizations, the second array can only be 2 GiB
    if in2.nbytes >= (1 << 31):
        raise RuntimeError('smaller array must be 2 GiB or less, '
                           'use method="fft" instead')

    # At this point, in1.size > in2.size
    # (except some cases when boundary != 'constant' or fillvalue != 0)
    # Figure out the output shape and the origin of the kernel
    if mode == 'full':
        out_shape = tuple(x1+x2-1 for x1, x2 in zip(in1.shape, in2.shape))
        offsets = tuple(x-1 for x in in2.shape)
    elif mode == 'valid':
        out_shape = tuple(x1-x2+1 for x1, x2 in zip(in1.shape, in2.shape))
        offsets = (0,) * in1.ndim
    else:  # mode == 'same':
        # In correlate2d: When using "same" mode with even-length inputs, the
        # outputs of correlate and correlate2d differ: There is a 1-index
        # offset between them.
        # This is dealt with by using "shift" parameter.
        out_shape = orig_in1_shape
        if orig_in1_shape == in1.shape:
            offsets = tuple((x-shift)//2 for x in in2.shape)
        else:
            offsets = tuple((2*x2-x1-(not convolution)+shift)//2
                            for x1, x2 in zip(in1.shape, in2.shape))

    # Check the output
    if not isinstance(output, cupy.ndarray):
        output = cupy.empty(out_shape, output)
    elif output.shape != out_shape:
        raise ValueError("out has wrong shape")

    # Get and run the CuPy kernel
    int_type = _util._get_inttype(in1)
    kernel = filters._get_correlate_kernel(
        boundary, in2.shape, int_type, offsets, fillvalue)
    in2 = _reverse_and_conj(in2) if convolution else in2
    if not swapped_inputs:
        kernel(in1, in2, output)
    elif output.dtype.kind != 'c':
        # Avoids one array copy
        kernel(in1, in2, _reverse_and_conj(output))
    else:
        kernel(in1, in2, output)
        output = cupy.ascontiguousarray(_reverse_and_conj(output))
    return output


def _reverse_and_conj(x):
    # Reverse array `x` in all dimensions and perform the complex conjugate
    return x[(slice(None, None, -1),) * x.ndim].conj()


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    # See scipy's documentation in scipy.signal.signaltools
    if mode != 'valid' or not shape1:
        return False
    if axes is None:
        axes = range(len(shape1))
    not_ok1 = any(shape1[i] < shape2[i] for i in axes)
    not_ok2 = any(shape1[i] > shape2[i] for i in axes)
    if not_ok1 and not_ok2:
        raise ValueError("For 'valid' mode, one must be at least "
                         "as large as the other in every dimension")
    return not_ok1
