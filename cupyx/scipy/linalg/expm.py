import cupy
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import _util


def expm(a):
    """
    This method calculates the exponential of a square matrix, `` /exp{a}``.

    Args:
        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch
            of symmetric 2-D square matrices ``(..., M, M)``.

    Returns:
        scalar :class:`~cupy.ndarray`:

    .. seealso:: :func:`numpy.linalg.expm`
    """

    a = cupy.asarray(a)
    _util._assert_2d(a)
    _util._assert_stacked_square(a)

    # dtype = a.dtype

    # if dtype.char == 'f':
    #     getrf = cusolver.sgetrf
    #     getrf_bufferSize = cusolver.sgetrf_bufferSize
    # elif dtype.char == 'd':
    #     getrf = cusolver.dgetrf
    #     getrf_bufferSize = cusolver.dgetrf_bufferSize
    # elif dtype.char == 'F':
    #     getrf = cusolver.cgetrf
    #     getrf_bufferSize = cusolver.cgetrf_bufferSize
    # elif dtype.char == 'D':
    #     getrf = cusolver.zgetrf
    #     getrf_bufferSize = cusolver.zgetrf_bufferSize
    # else:
    #     msg = 'Only float32, float64, complex64
    # and complex128 are supported.'
    #     raise NotImplementedError(msg)

    pass
