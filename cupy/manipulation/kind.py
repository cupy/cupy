import numpy

import cupy


# TODO(okuta): Implement asfarray


def asfortranarray(a, dtype=None):
    """Return an array laid out in Fortran order in memory.

    Args:
        a (~cupy.ndarray): The input array.
        dtype (str or dtype object, optional): By default, the data-type is
            inferred from the input data.

    Returns:
        ~cupy.ndarray: The input `a` in Fortran, or column-major, order.

    .. seealso:: :func:`numpy.asfortranarray`

    """
    ret = cupy.empty(a.shape[::-1], a.dtype if dtype is None else dtype).T
    if (a.flags.c_contiguous and
            (a.dtype == numpy.float32 or a.dtype == numpy.float64) and
            a.ndim == 2 and
            dtype is None):
        m, n = a.shape
        if a.dtype == numpy.float32:
            cupy.cuda.cublas.sgeam(
                cupy.cuda.Device().cublas_handle,
                1,  # transpose a
                1,  # transpose ret
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                ret.data.ptr, m)
        elif a.dtype == numpy.float64:
            cupy.cuda.cublas.dgeam(
                cupy.cuda.Device().cublas_handle,
                1,  # transpose a
                1,  # transpose ret
                m, n, 1., a.data.ptr, n, 0., a.data.ptr, n,
                ret.data.ptr, m)
        return ret
    else:
        ret[...] = a
        return ret


# TODO(okuta): Implement asarray_chkfinite


# TODO(okuta): Implement asscalar


# TODO(okuta): Implement require
