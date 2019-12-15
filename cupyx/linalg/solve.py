import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.linalg import util


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

    # to prevent `a` from being overwritten
    a = a.copy()

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

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
