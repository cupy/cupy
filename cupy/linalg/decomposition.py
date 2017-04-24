import numpy
from numpy import linalg

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def cholesky(a):
    '''Cholesky decomposition.

    Decompose a given two-dimensional square matrix into ``L * L.T``,
    where ``L`` is a lower-triangular matrix and ``.T`` is a conjugate
    transpose operator. Note that in the current implementation ``a`` must be
    a real matrix, and only float32 and float64 are supported.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(N, N)``

    .. seealso:: :func:`numpy.linalg.cholesky`
    '''
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    _assert_cupy_array(a)
    _assert_rank2(a)
    _assert_nd_squareness(a)

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    x = a.astype(dtype, copy=True)
    n = len(a)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    if dtype == 'f':
        buffersize = cusolver.spotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        cusolver.spotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, dev_info.data.ptr)
    else:  # dtype == 'd'
        buffersize = cusolver.dpotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        cusolver.dpotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, dev_info.data.ptr)
    status = int(dev_info[0])
    if status > 0:
        raise linalg.LinAlgError(
            'The leading minor of order {} '
            'is not positive definite'.format(status))
    elif status < 0:
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')
    _tril(x, k=0)
    return x


# TODO(okuta): Implement qr


# TODO(okuta): Implement svd


def _assert_cupy_array(*arrays):
    for a in arrays:
        if not isinstance(a, cupy.core.ndarray):
            raise linalg.LinAlgError(
                'cupy.linalg only supports cupy.core.ndarray')


def _assert_rank2(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise linalg.LinAlgError(
                '{}-dimensional array given. Array must be '
                'two-dimensional'.format(a.ndim))


def _assert_nd_squareness(*arrays):
    for a in arrays:
        if max(a.shape[-2:]) != min(a.shape[-2:]):
            raise linalg.LinAlgError(
                'Last 2 dimensions of the array must be square')


def _tril(x, k=0):
    m, n = x.shape
    u = cupy.arange(m).reshape(m, 1)
    v = cupy.arange(n).reshape(1, n)
    mask = v - u <= k
    x *= mask
    return x
