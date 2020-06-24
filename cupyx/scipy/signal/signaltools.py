
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
    # TODO(Dahlia-Chehata): replace with GPU-based constants.
    return True
