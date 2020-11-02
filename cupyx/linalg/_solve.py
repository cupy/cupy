import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cusolver import check_availability
from cupy.linalg import _util


def _batched_invh(a):
    """Compute the inverse of an array of Hermitian matrices.

    This function computes an inverse of a real symmetric or complex hermitian
    positive-definite matrix using Cholesky factorization. If matrix ``a[i]``
    is not positive definite, Cholesky factorization fails and it raises
    an error.

    Args:
        a (cupy.ndarray): Array of real symmetric or complex hermitian
            matrices with dimension (..., N, N).

    Returns:
        cupy.ndarray: The array of inverses of matrices ``a[i]``.
    """
    if not check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f').char

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
    n = a.shape[-1]
    lda = a.strides[-2] // a.dtype.itemsize
    handle = device.get_cusolver_handle()
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    batch_size = int(numpy.prod(a.shape[:-2]))
    dev_info = cupy.empty(batch_size, dtype=numpy.int32)

    # Cholesky factorization
    potrfBatched(handle, uplo, n, ap.data.ptr, lda, dev_info.data.ptr,
                 batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

    identity_matrix = cupy.eye(n, dtype=dtype)
    b = cupy.empty(a.shape, dtype)
    b[...] = identity_matrix
    nrhs = b.shape[-1]
    ldb = b.strides[-2] // a.dtype.itemsize
    bp = cupy.core._mat_ptrs(b)
    dev_info = cupy.empty(1, dtype=numpy.int32)

    # NOTE: potrsBatched does not currently support nrhs > 1 (CUDA v10.2)
    # Solve: A[i] * X[i] = B[i]
    potrsBatched(handle, uplo, n, nrhs, ap.data.ptr, lda, bp.data.ptr, ldb,
                 dev_info.data.ptr, batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

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
    if a.ndim > 2:
        return _batched_invh(a)

    # to prevent `a` from being overwritten
    a = a.copy()

    # support float32, float64, complex64, and complex128
    if a.dtype.char in 'fdFD':
        dtype = a.dtype.char
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f').char

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

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

    m = a.shape[0]
    uplo = cublas.CUBLAS_FILL_MODE_LOWER

    worksize = potrf_bufferSize(cusolver_handle, uplo, m, a.data.ptr, m)
    workspace = cupy.empty(worksize, dtype=dtype)

    # Cholesky factorization
    potrf(cusolver_handle, uplo, m, a.data.ptr, m, workspace.data.ptr,
          worksize, dev_info.data.ptr)

    info = dev_info[0]
    if info != 0:
        if info < 0:
            msg = '\tThe {}-th parameter is wrong'.format(-info)
        else:
            msg = ('\tThe leading minor of order {} is not positive definite'
                   .format(info))
        raise RuntimeError('matrix inversion failed at potrf.\n' + msg)

    b = cupy.eye(m, dtype=dtype)

    # Solve: A * X = B
    potrs(cusolver_handle, uplo, m, m, a.data.ptr, m, b.data.ptr, m,
          dev_info.data.ptr)

    info = dev_info[0]
    if info > 0:
        assert False, ('Unexpected output returned by potrs (actual: {})'
                       .format(info))
    elif info < 0:
        raise RuntimeError('matrix inversion failed at potrs.\n'
                           '\tThe {}-th parameter is wrong'.format(-info))

    return b
