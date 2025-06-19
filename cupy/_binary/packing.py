import cupy
from cupy import _core


_packbits_kernel = {
    'big': _core.ElementwiseKernel(
        'raw T a, raw int32 a_size', 'uint8 packed',
        '''for (int j = 0; j < 8; ++j) {
                    int k = i * 8 + j;
                    int bit = k < a_size && a[k] != 0;
                    packed |= bit << (7 - j);
                }''',
        'cupy_packbits_big'
    ),
    'little': _core.ElementwiseKernel(
        'raw T a, raw int32 a_size', 'uint8 packed',
        '''for (int j = 0; j < 8; ++j) {
                    int k = i * 8 + j;
                    int bit = k < a_size && a[k] != 0;
                    packed |= bit << j;
                }''',
        'cupy_packbits_little'
    )
}


def packbits(a, axis=None, bitorder='big'):
    """Packs the elements of a binary-valued array into bits in a uint8 array.

    This function currently does not support ``axis`` option.

    Args:
        a (cupy.ndarray): Input array.
        axis (int, optional): Not supported yet.
        bitorder (str, optional): bit order to use when packing the array,
            allowed values are `'little'` and `'big'`. Defaults to `'big'`.

    Returns:
        cupy.ndarray: The packed array.

    .. note::
        When the input array is empty, this function returns a copy of it,
        i.e., the type of the output array is not necessarily always uint8.
        This exactly follows the NumPy's behaviour (as of version 1.11),
        alghough this is inconsistent to the documentation.

    .. seealso:: :func:`numpy.packbits`
    """
    if a.dtype.kind not in 'biu':
        raise TypeError(
            'Expected an input array of integer or boolean data type')

    if axis is not None:
        raise NotImplementedError('axis option is not supported yet')

    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'big' or 'little'")

    a = a.ravel()
    packed_size = (a.size + 7) // 8
    packed = cupy.zeros((packed_size,), dtype=cupy.uint8)
    return _packbits_kernel[bitorder](a, a.size, packed)


_unpackbits_kernel = {
    'big': _core.ElementwiseKernel(
        'raw uint8 a', 'T unpacked',
        'unpacked = (a[i / 8] >> (7 - i % 8)) & 1;',
        'cupy_unpackbits_big'
    ),
    'little': _core.ElementwiseKernel(
        'raw uint8 a', 'T unpacked',
        'unpacked = (a[i / 8] >> (i % 8)) & 1;',
        'cupy_unpackbits_little'
    )
}


def unpackbits(a, axis=None, bitorder='big'):
    """Unpacks elements of a uint8 array into a binary-valued output array.

    This function currently does not support ``axis`` option.

    Args:
        a (cupy.ndarray): Input array.
        bitorder (str, optional): bit order to use when unpacking the array,
            allowed values are `'little'` and `'big'`. Defaults to `'big'`.

    Returns:
        cupy.ndarray: The unpacked array.

    .. seealso:: :func:`numpy.unpackbits`
    """
    if a.dtype != cupy.uint8:
        raise TypeError('Expected an input array of unsigned byte data type')

    if axis is not None:
        raise NotImplementedError('axis option is not supported yet')

    if bitorder not in ('big', 'little'):
        raise ValueError("bitorder must be either 'big' or 'little'")

    unpacked = cupy.ndarray((a.size * 8), dtype=cupy.uint8)
    return _unpackbits_kernel[bitorder](a, unpacked)
