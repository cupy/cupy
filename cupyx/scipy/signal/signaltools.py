import cupy


def _numeric_arrays(arrays, kinds='buifc'):
    if type(arrays) == cupy.ndarray:
        return arrays.dtype.kind in kinds
    for array in arrays:
        if array.dtype.kind not in kinds:
            return False
    return True


def choose_conv_method(in1, in2, mode='full'):
    """Find the fastest convolution/correlation method.

    Args:
        in1 (cupy.ndarray): first input.
        in2 (cupy.ndarray): second input.
        mode (str, optional): `valid`, `same`, `full`

    Returns:
        str: A string indicating which convolution method is fastest,
         either ‘direct’ or ‘fft’.

    .. warning::
        This function currently doesn't support measure option,
        nor multidimensional inputs.

    .. seealso:: :func:`scipy.signal.choose_conv_method`

    """
    if in1.ndim != 1 or in2.ndim != 1:
        raise NotImplementedError('Only 1d inputs are supported currently')

    if any([_numeric_arrays([x], kinds='ui') for x in [in1, in2]]):
        max_value = int(cupy.abs(in1).max()) * int(cupy.abs(in2).max())
        max_value *= int(min(in1.size, in2.size))
        if max_value > 2 ** cupy.finfo('float').nmant - 1:
            return 'direct'

    if _numeric_arrays([in1, in2], kinds='b'):
        return 'direct'

    if _numeric_arrays([in1, in2]):
        if _fftconv_faster(in1, in2, mode):
            return 'fft'

    return 'direct'


def _fftconv_faster(x, h, mode):
    fft_ops, direct_ops = _conv_ops(x.size, h.size, mode)
    offset = -1e-3
    constants = {
        "valid": (1.89095737e-9, 2.1364985e-10, offset),
        "full": (1.7649070e-9, 2.1414831e-10, offset),
        "same": (3.2646654e-9, 2.8478277e-10, offset)
        if h.size <= x.size
        else (3.21635404e-9, 1.1773253e-8, -1e-5),
    }
    O_fft, O_direct, O_offset = constants[mode]
    return O_fft * fft_ops < O_direct * direct_ops + O_offset


def _conv_ops(siz1, siz2, mode):
    if mode == "full":
        direct_ops = siz1 * siz2
    elif mode == "valid":
        direct_ops = (siz2 - siz1 + 1) * siz1 \
            if siz2 >= siz1 else (siz1 - siz2 + 1) * siz2
    elif mode == "same":
        direct_ops = siz1 * siz2 if siz1 < siz2\
            else siz1 * siz2 - (siz2 / 2) * ((siz2 + 1) / 2)
    else:
        raise ValueError('Unsupported mode')
    N = siz1 + siz2 - 1
    fft_ops = 3 * N * cupy.log(N)
    return fft_ops, direct_ops
