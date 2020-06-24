import math


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
        the compatibility of the return value to NumPy's one.

    .. seealso:: :func:`scipy.signal.choose_conv_method`

    """
    if in1.ndim != 1 or in2.ndim != 1:
        raise NotImplementedError('Only 1d inputs are supported currently')

    if in1.dtype.kind in 'bui' or in2.dtype.kind in 'bui':
        return 'direct'

    if _fftconv_faster(in1, in2, mode):
        return 'fft'

    return 'direct'


def _fftconv_faster(x, h, mode):
    """
    .. seealso:: :func: `scipy.signal.signaltools._fftconv_faster`

    """
    fft_ops, direct_ops = _conv_ops(x.size, h.size, mode)
    # TODO(Dahlia-Chehata): replace with GPU-based constants.
    return True


def _conv_ops(siz1, siz2, mode):
    if mode == 'full':
        direct_ops = siz1 * siz2
    elif mode == 'valid':
        if siz2 >= siz1:
            direct_ops = (siz2 - siz1 + 1) * siz1
        else:
            direct_ops = (siz1 - siz2 + 1) * siz2
    elif mode == 'same':
        if siz1 < siz2:
            direct_ops = siz1 * siz2
        else:
            direct_ops = siz1 * siz2 - (siz2 / 2) * ((siz2 + 1) / 2)
    else:
        raise ValueError('Unsupported mode')
    N = siz1 + siz2 - 1
    fft_ops = 3 * N * math.log(N)
    return fft_ops, direct_ops
