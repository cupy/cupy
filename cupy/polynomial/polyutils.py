import cupy


def trimseq(seq):
    """Removes small polynomial series coefficients.

    Args:
        seq (cupy.ndarray): input array.

    Returns:
        cupy.ndarray: input array with trailing zeros removed. If the
            resulting output is empty, it returns the first element

    .. seealso:: :func:`numpy.polynomial.polyutils.trimseq`

    """
    if seq.size == 0:
        return seq
    ret = cupy.trim_zeros(seq, trim='b')
    if ret.size > 0:
        return ret
    return seq[:1]


def as_series(alist, trim=True):
    """Returns argument as a list of 1-d arrays.

    Args:
        alist (cupy.ndarray or list of cupy.ndarray): 1-D or 2-D input array.
        trim (bool, optional): trim trailing zeros.

    Returns:
        list of cupy.ndarray: list of 1-D arrays.

    .. seealso:: :func:`numpy.polynomial.polyutils.as_series`

    """
    arrays = []
    for a in alist:
        a = cupy.array(a, ndmin=1, copy=False)
        if not a.size:
            raise ValueError('Coefficient array is empty')
        if a.ndim != 1:
            raise ValueError('Coefficient array is not 1-d')
        if trim:
            a = trimseq(a)
        arrays.append(a)
    try:
        dtype = cupy.common_type(*arrays)
    except Exception:
        raise ValueError('Coefficient arrays have no common type')
    ret = [a.astype(dtype, copy=False) for a in arrays]
    return ret
