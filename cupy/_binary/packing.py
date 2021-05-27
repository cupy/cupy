import cupy
from cupy import _core


_packbits_kernel = _core.ElementwiseKernel(
    'raw T myarray, raw int32 myarray_size', 'uint8 packed',
    '''for (int j = 0; j < 8; ++j) {
        int k = i * 8 + j;
        int bit = k < myarray_size && myarray[k] != 0;
        packed |= bit << (7 - j);
    }''',
    'packbits_kernel'
)


def packbits(myarray):
    """Packs the elements of a binary-valued array into bits in a uint8 array.

    This function currently does not support ``axis`` option.

    Args:
        myarray (cupy.ndarray): Input array.

    Returns:
        cupy.ndarray: The packed array.

    .. note::
        When the input array is empty, this function returns a copy of it,
        i.e., the type of the output array is not necessarily always uint8.
        This exactly follows the NumPy's behaviour (as of version 1.11),
        alghough this is inconsistent to the documentation.

    .. seealso:: :func:`numpy.packbits`
    """
    if myarray.dtype.kind not in 'biu':
        raise TypeError(
            'Expected an input array of integer or boolean data type')

    myarray = myarray.ravel()
    packed_size = (myarray.size + 7) // 8
    packed = cupy.zeros((packed_size,), dtype=cupy.uint8)
    return _packbits_kernel(myarray, myarray.size, packed)


_unpackbits_kernel = _core.ElementwiseKernel(
    'raw uint8 myarray', 'T unpacked',
    'unpacked = (myarray[i / 8] >> (7 - i % 8)) & 1;',
    'unpackbits_kernel'
)


def unpackbits(myarray):
    """Unpacks elements of a uint8 array into a binary-valued output array.

    This function currently does not support ``axis`` option.

    Args:
        myarray (cupy.ndarray): Input array.

    Returns:
        cupy.ndarray: The unpacked array.

    .. seealso:: :func:`numpy.unpackbits`
    """
    if myarray.dtype != cupy.uint8:
        raise TypeError('Expected an input array of unsigned byte data type')

    unpacked = cupy.ndarray((myarray.size * 8), dtype=cupy.uint8)
    return _unpackbits_kernel(myarray, unpacked)
