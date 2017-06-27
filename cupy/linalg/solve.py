import numpy
from numpy import linalg

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg.util import _assert_cupy_array
from cupy.linalg.util import _assert_nd_squareness
from cupy.linalg.util import _assert_rank2

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def solve(a, b):
    '''Solves a linear matrix equation.

    It computes the exact solution of ``x`` in ``ax = b``,
    where ``a`` is a square and full rank matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``
        b (cupy.ndarray): The vector with ``M`` elements, or
            the matrix with dimension ``(M, K)``

    .. seealso:: :func:`numpy.linalg.solve`
    '''
    # NOTE: Since cusolver in CUDA 8.0 does not support gesv,
    #       we manually solve a linear system with QR decomposition.
    #       For details, please see the following:
    #       http://docs.nvidia.com/cuda/cusolver/index.html#qr_examples
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    _assert_cupy_array(a, b)
    _assert_rank2(a)
    _assert_nd_squareness(a)
    if 2 < b.ndim:
        raise linalg.LinAlgError(
            '{}-dimensional array given. Array must be '
            'one or two-dimensional'.format(b.ndim))
    if a.shape[0] != b.shape[0]:
        raise linalg.LinAlgError(
            'The number of rows of array a must be '
            'the same as that of array b')

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    m, k = (b.size, 1) if b.ndim == 1 else b.shape
    a = a.transpose().astype(dtype, order='C', copy=True)
    b = b.transpose().astype(dtype, order='C', copy=True)
    cusolver_handle = device.get_cusolver_handle()
    cublas_handle = device.get_cublas_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        geqrf = cusolver.sgeqrf
        geqrf_bufferSize = cusolver.sgeqrf_bufferSize
        ormqr = cusolver.sormqr
        trsm = cublas.strsm
    else:  # dtype == 'd'
        geqrf = cusolver.dgeqrf
        geqrf_bufferSize = cusolver.dgeqrf_bufferSize
        ormqr = cusolver.dormqr
        trsm = cublas.dtrsm

    # 1. QR decomposition (A = Q * R)
    buffersize = geqrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)
    tau = cupy.empty(m, dtype=dtype)
    geqrf(
        cusolver_handle, m, m, a.data.ptr, m,
        tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)
    _check_status(dev_info)
    # 2. ormqr (Q^T * B)
    ormqr(
        cusolver_handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_OP_T,
        m, k, m, a.data.ptr, m, tau.data.ptr, b.data.ptr, m,
        workspace.data.ptr, buffersize, dev_info.data.ptr)
    _check_status(dev_info)
    # 3. trsm (X = R^{-1} * (Q^T * B))
    trsm(
        cublas_handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
        cublas.CUBLAS_OP_N, cublas.CUBLAS_DIAG_NON_UNIT,
        m, k, 1, a.data.ptr, m, b.data.ptr, m)
    return b.transpose()


def _check_status(dev_info):
    status = int(dev_info[0])
    if status < 0:
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')


# TODO(okuta): Implement tensorsolve


# TODO(okuta): Implement lstsq


# TODO(okuta): Implement inv


# TODO(okuta): Implement pinv


# TODO(okuta): Implement tensorinv
