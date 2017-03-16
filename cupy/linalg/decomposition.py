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


def qr(a, mode='reduced'):
    '''QR decomposition.

    Decompose a given two-dimensional matrix into ``Q * R``, where ``Q``
    is an orthonormal and ``R`` is an upper-triangular matrix.

    Args:
        a (cupy.ndarray): The input matrix.
        mode (str): The mode of decomposition. Currently 'reduced',
            'complete', 'r', and 'raw' modes are supported. The default mode
            is 'reduced', and decompose a matrix ``A = (M, N)`` into ``Q``,
            ``R`` with dimensions ``(M, K)``, ``(K, N)``, where
            ``K = min(M, N)``.

    .. seealso:: :func:`numpy.linalg.qr`
    '''
    if not cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    _assert_cupy_array(a)
    _assert_rank2(a)

    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full', 'e', 'economic'):
            msg = 'The deprecated mode \'{}\' is not supported'.format(mode)
            raise ValueError(msg)
        else:
            raise ValueError('Unrecognized mode \'{}\''.format(mode))

    ret_dtype = a.dtype.char
    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    m, n = a.shape
    x = a.transpose().astype(dtype, copy=True)
    mn = min(m, n)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    # compute working space of geqrf and ormqr, and solve R
    if x.dtype.char == 'f':
        buffersize = cusolver.sgeqrf_bufferSize(handle, m, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        tau = cupy.empty(mn, dtype=numpy.float32)
        cusolver.sgeqrf(
            handle, m, n, x.data.ptr, m,
            tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)
    else:  # a.dtype.char == 'd'
        buffersize = cusolver.dgeqrf_bufferSize(handle, n, m, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        tau = cupy.empty(mn, dtype=numpy.float64)
        cusolver.dgeqrf(
            handle, m, n, x.data.ptr, m,
            tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)
    status = int(dev_info[0])
    if status < 0:
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')

    if mode == 'r':
        r = x[:, :mn].transpose().astype(dtype, copy=True)
        return _triu(r)

    if mode == 'raw':
        if ret_dtype == 'f':
            # The original numpy.linalg.qr returns float64 in raw mode,
            # whereas the cusolver returns float32. We agree that the
            # following code would be inappropriate, however, in this time
            # we explicitly convert them to float64 for compatibility.
            return x.astype(numpy.float64), tau.astype(numpy.float64)
        return x, tau

    if mode == 'complete' and m > n:
        mc = m
        q = cupy.empty((m, m), dtype)
    else:
        mc = mn
        q = cupy.empty((n, m), dtype)
    q[:n] = x

    # solve Q
    if x.dtype.char == 'f':
        buffersize = cusolver.sorgqr_bufferSize(
            handle, m, mc, mn, q.data.ptr, m, tau.data.ptr)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        cusolver.sorgqr(
            handle, m, mc, mn, q.data.ptr, m, tau.data.ptr,
            workspace.data.ptr, buffersize, dev_info.data.ptr)
    else:
        buffersize = cusolver.dorgqr_bufferSize(
            handle, m, mc, mn, q.data.ptr, m, tau.data.ptr)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        cusolver.dorgqr(
            handle, m, mc, mn, q.data.ptr, m, tau.data.ptr,
            workspace.data.ptr, buffersize, dev_info.data.ptr)

    q = q[:mc].transpose().astype(dtype, copy=True)
    r = x[:, :mc].transpose().astype(dtype, copy=True)
    return q, _triu(r)


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


def _triu(x, k=0):
    m, n = x.shape
    u = cupy.arange(m).reshape(m, 1)
    v = cupy.arange(n).reshape(1, n)
    mask = v - u >= k
    x *= mask
    return x
