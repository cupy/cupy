import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cusolver import check_availability
from cupy.linalg import _util


def _batched_cholesky_solve(a, b):

    if not check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrfBatched = cusolver.spotrfBatched
        potrsBatched = cusolver.spotrsBatched
    elif dtype == 'd':
        potrfBatched = cusolver.dpotrfBatched
        potrsBatched = cusolver.dpotrsBatched
    elif dtype == 'F':
        potrfBatched = cusolver.cpotrfBatched
        potrsBatched = cusolver.cpotrsBatched
    elif dtype == 'D':
        potrfBatched = cusolver.zpotrfBatched
        potrsBatched = cusolver.zpotrsBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    a = a.astype(dtype, order='C', copy=True)
    ap = cupy.core._mat_ptrs(a)
    lda, n = a.shape[-2:]
    batch_size = int(numpy.prod(a.shape[:-2]))

    handle = device.get_cusolver_handle()
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = cupy.empty(batch_size, dtype=numpy.int32)

    # Cholesky factorization
    potrfBatched(handle, uplo, n, ap.data.ptr, lda, dev_info.data.ptr,
                 batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

    b_shape = b.shape
    b = b.reshape(batch_size, n, -1).astype(dtype, order='C', copy=True)
    bp = cupy.core._mat_ptrs(b)
    ldb, nrhs = b.shape[-2:]
    dev_info = cupy.empty(1, dtype=numpy.int32)

    # NOTE: potrsBatched does not currently support nrhs > 1 (CUDA v10.2)
    # Solve: A[i] * X[i] = B[i]
    potrsBatched(handle, uplo, n, nrhs, ap.data.ptr, lda, bp.data.ptr, ldb,
                 dev_info.data.ptr, batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrsBatched, dev_info)

    return b.conj().reshape(b_shape)


def cholesky_solve(a, b):
    """Solve the linear equations A x = b via Cholesky factorization of A,
    where A is a real symmetric or complex Hermitian positive-definite matrix.

    If matrix ``a[i]`` is not positive definite, Cholesky factorization fails
    and it raises an error.

    Args:
        a (cupy.ndarray): Array of real symmetric or complex hermitian
            matrices with dimension (..., N, N).
        b (cupy.ndarray): right-hand side (..., N).
    Returns:
        x (cupy.ndarray): The array of solutions ``x[i]``.
    """

    _util._assert_cupy_array(a, b)
    _util._assert_nd_squareness(a)

    if a.ndim > 2:
        return _batched_cholesky_solve(a, b)

    dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
        potrs = cusolver.spotrs
    elif dtype == 'd':
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
        potrs = cusolver.dpotrs
    elif dtype == 'F':
        potrf = cusolver.cpotrf
        potrf_bufferSize = cusolver.cpotrf_bufferSize
        potrs = cusolver.cpotrs
    elif dtype == 'D':
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize
        potrs = cusolver.zpotrs
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    a = a.astype(dtype, order='C', copy=True)
    lda, n = a.shape

    handle = device.get_cusolver_handle()
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = cupy.empty(1, dtype=numpy.int32)

    worksize = potrf_bufferSize(handle, uplo, n, a.data.ptr, lda)
    workspace = cupy.empty(worksize, dtype=dtype)

    # Cholesky factorization
    potrf(handle, uplo, n, a.data.ptr, lda, workspace.data.ptr,
          worksize, dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    b_shape = b.shape
    b = b.reshape(n, -1).astype(dtype, order='C', copy=True)
    ldb, nrhs = b.shape

    # Solve: A * X = B
    potrs(handle, uplo, n, nrhs, a.data.ptr, lda, b.data.ptr, ldb,
          dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrs, dev_info)

    b = b.reshape(b_shape)

    if nrhs == 1:
        return b.conj()
    else:
        return b


def invh(a):
    """Compute the inverse of a Hermitian matrix.

    This function computes a inverse of a real symmetric or complex hermitian
    positive-definite matrix using Cholesky factorization. If matrix ``a`` is
    not positive definite, Cholesky factorization fails and it raises an error.

    Args:
        a (cupy.ndarray): Real symmetric or complex hermitian maxtix.

    Returns:
        cupy.ndarray: The inverse of matrix ``a``.
    """

    _util._assert_cupy_array(a)
    _util._assert_nd_squareness(a)

    # TODO: Remove this assert once cusolver supports nrhs > 1 for potrsBatched
    _util._assert_rank2(a)

    n = a.shape[-1]
    identity_matrix = cupy.eye(n, dtype=a.dtype)
    b = cupy.empty(a.shape, a.dtype)
    b[...] = identity_matrix

    return cholesky_solve(a, b)
