import cupy
from cupy.core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import filters
from cupyx.scipy.ndimage import _util


def _check_conv_inputs(in1, in2, mode, convolution=True):
    if in1.ndim == in2.ndim == 0:
        return in1 * (in2 if convolution else in2.conj())
    if in1.ndim != in2.ndim:
        raise ValueError('in1 and in2 should have the same dimensionality')
    if in1.size == 0 or in2.size == 0:
        return cupy.array([], dtype=in1.dtype)
    if mode not in ('full', 'same', 'valid'):
        raise ValueError('acceptable modes are "valid", "same", or "full"')
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
        raise ValueError('out has wrong shape')

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
        raise ValueError('For "valid" mode, one must be at least '
                         'as large as the other in every dimension')
    return not_ok1


def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    # See scipy's documentation in scipy.signal.signaltools
    s1, s2 = in1.shape, in2.shape
    axes = _init_nd_and_axes(in1, axes)
    # Length-1 axes can rely on broadcasting rules, no fft needed
    axes = [ax for ax in axes if s1[ax] != 1 and s2[ax] != 1]
    if sorted_axes:
        axes.sort()

    # Check that unused axes are either 1 (broadcast) or the same length
    for ax, (dim1, dim2) in enumerate(zip(s1, s2)):
        if ax not in axes and dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise ValueError('incompatible shapes for in1 and in2:'
                             ' {} and {}'.format(s1, s2))

    # Check that input sizes are compatible with 'valid' mode.
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # Convolution is commutative
        in1, in2 = in2, in1

    return in1, in2, axes


def _init_nd_and_axes(x, axes):
    # See documentation in scipy.fft._helper._init_nd_shape_and_axes
    # except shape argument is always None and doesn't return new shape
    axes = internal._normalize_axis_indices(axes, x.ndim, sort_axes=False)
    if not len(axes):
        raise ValueError('when provided, axes cannot be empty')
    if any(x.shape[ax] < 1 for ax in axes):
        raise ValueError('invalid number of data points specified')
    return axes


def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    # See scipy's documentation in scipy.signal.signaltools
    real = (in1.dtype.kind != 'c' and in2.dtype.kind != 'c')
    fshape = ([fft.next_fast_len(shape[a], real) for a in axes]
              if calc_fast_len else shape)
    fftn, ifftn = (fft.rfftn, fft.irfftn) if real else (fft.fftn, fft.ifftn)

    # Perform the convolution
    sp1 = fftn(in1, fshape, axes=axes)
    sp2 = fftn(in2, fshape, axes=axes)
    out = ifftn(sp1 * sp2, fshape, axes=axes)

    return out[tuple(slice(x) for x in shape)] if calc_fast_len else out


def _apply_conv_mode(full, s1, s2, mode, axes):
    # See scipy's documentation in scipy.signal.signaltools
    if mode == 'full':
        return cupy.ascontiguousarray(full)
    if mode == 'valid':
        s1 = [full.shape[a] if a not in axes else s1[a] - s2[a] + 1
              for a in range(full.ndim)]
    starts = [(cur-new)//2 for cur, new in zip(full.shape, s1)]
    slices = tuple(slice(start, start+length)
                   for start, length in zip(starts, s1))
    return cupy.ascontiguousarray(full[slices])
