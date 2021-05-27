import numpy as _numpy

import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device


def gesv(a, b):
    """Solve a linear matrix equation using cusolverDn<t>getr[fs]().

    Computes the solution to a system of linear equation ``ax = b``.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M)`` or ``(M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M)`` or ``(M, K)``.

    Note: ``a`` and ``b`` will be overwritten.
    """
    if a.ndim != 2:
        raise ValueError('a.ndim must be 2 (actual: {})'.format(a.ndim))
    if b.ndim not in (1, 2):
        raise ValueError('b.ndim must be 1 or 2 (actual: {})'.format(b.ndim))
    if a.shape[0] != a.shape[1]:
        raise ValueError('a must be a square matrix.')
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape mismatch (a: {}, b: {}).'.
                         format(a.shape, b.shape))
    if a.dtype != b.dtype:
        raise TypeError('dtype mismatch (a: {}, b: {})'.
                        format(a.dtype, b.dtype))
    dtype = a.dtype
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise TypeError('unsupported dtype (actual:{})'.format(a.dtype))
    helper = getattr(_cusolver, t + 'getrf_bufferSize')
    getrf = getattr(_cusolver, t + 'getrf')
    getrs = getattr(_cusolver, t + 'getrs')

    n = b.shape[0]
    nrhs = b.shape[1] if b.ndim == 2 else 1
    if a._f_contiguous:
        trans = _cublas.CUBLAS_OP_N
    elif a._c_contiguous:
        trans = _cublas.CUBLAS_OP_T
    else:
        raise ValueError('a must be F-contiguous or C-contiguous.')
    if not b._f_contiguous:
        raise ValueError('b must be F-contiguous.')

    handle = _device.get_cusolver_handle()
    dipiv = _cupy.empty(n, dtype=_numpy.int32)
    dinfo = _cupy.empty(1, dtype=_numpy.int32)
    lwork = helper(handle, n, n, a.data.ptr, n)
    dwork = _cupy.empty(lwork, dtype=a.dtype)
    # LU factrization (A = L * U)
    getrf(handle, n, n, a.data.ptr, n, dwork.data.ptr, dipiv.data.ptr,
          dinfo.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        getrf, dinfo)
    # Solves Ax = b
    getrs(handle, trans, n, nrhs, a.data.ptr, n,
          dipiv.data.ptr, b.data.ptr, n, dinfo.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        getrs, dinfo)


def gels(a, b):
    """Solves over/well/under-determined linear systems.

    Computes least-square solution to equation ``ax = b` by QR factorization
    using cusolverDn<t>geqrf().

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, N)``.
        b (cupy.ndarray): The matrix with dimension ``(M)`` or ``(M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(N)`` or ``(N, K)``.
    """
    if a.ndim != 2:
        raise ValueError('a.ndim must be 2 (actual: {})'.format(a.ndim))
    if b.ndim == 1:
        nrhs = 1
    elif b.ndim == 2:
        nrhs = b.shape[1]
    else:
        raise ValueError('b.ndim must be 1 or 2 (actual: {})'.format(b.ndim))
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape mismatch (a: {}, b: {}).'.
                         format(a.shape, b.shape))
    if a.dtype != b.dtype:
        raise ValueError('dtype mismatch (a: {}, b: {}).'.
                         format(a.dtype, b.dtype))

    dtype = a.dtype
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise ValueError('unsupported dtype (actual: {})'.format(dtype))

    geqrf_helper = getattr(_cusolver, t + 'geqrf_bufferSize')
    geqrf = getattr(_cusolver, t + 'geqrf')
    trsm = getattr(_cublas, t + 'trsm')
    if t in 'sd':
        ormqr_helper = getattr(_cusolver, t + 'ormqr_bufferSize')
        ormqr = getattr(_cusolver, t + 'ormqr')
    else:
        ormqr_helper = getattr(_cusolver, t + 'unmqr_bufferSize')
        ormqr = getattr(_cusolver, t + 'unmqr')

    no_trans = _cublas.CUBLAS_OP_N
    if dtype.char in 'fd':
        trans = _cublas.CUBLAS_OP_T
    else:
        trans = _cublas.CUBLAS_OP_C

    m, n = a.shape
    mn_min = min(m, n)
    dev_info = _cupy.empty(1, dtype=_numpy.int32)
    tau = _cupy.empty(mn_min, dtype=dtype)
    cusolver_handle = _device.get_cusolver_handle()
    cublas_handle = _device.get_cublas_handle()
    one = _numpy.array(1.0, dtype=dtype)

    if m >= n:  # over/well-determined systems
        a = a.copy(order='F')
        b = b.copy(order='F')

        # geqrf (QR decomposition, A = Q * R)
        ws_size = geqrf_helper(cusolver_handle, m, n, a.data.ptr, m)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        geqrf(cusolver_handle, m, n, a.data.ptr, m, tau.data.ptr,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            geqrf, dev_info)

        # ormqr (Computes Q^T * B)
        ws_size = ormqr_helper(
            cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, trans, m, nrhs, mn_min,
            a.data.ptr, m, tau.data.ptr, b.data.ptr, m)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        ormqr(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, trans, m, nrhs,
              mn_min, a.data.ptr, m, tau.data.ptr, b.data.ptr, m,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            ormqr, dev_info)

        # trsm (Solves R * X = (Q^T * B))
        trsm(cublas_handle, _cublas.CUBLAS_SIDE_LEFT,
             _cublas.CUBLAS_FILL_MODE_UPPER, no_trans,
             _cublas.CUBLAS_DIAG_NON_UNIT, mn_min, nrhs,
             one.ctypes.data, a.data.ptr, m, b.data.ptr, m)

        return b[:n]

    else:  # under-determined systems
        a = a.conj().T.copy(order='F')
        bb = b
        out_shape = (n,) if b.ndim == 1 else (n, nrhs)
        b = _cupy.zeros(out_shape, dtype=dtype, order='F')
        b[:m] = bb

        # geqrf (QR decomposition, A^T = Q * R)
        ws_size = geqrf_helper(cusolver_handle, n, m, a.data.ptr, n)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        geqrf(cusolver_handle, n, m, a.data.ptr, n, tau.data.ptr,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            geqrf, dev_info)

        # trsm (Solves R^T * Z = B)
        trsm(cublas_handle, _cublas.CUBLAS_SIDE_LEFT,
             _cublas.CUBLAS_FILL_MODE_UPPER, trans,
             _cublas.CUBLAS_DIAG_NON_UNIT, m, nrhs,
             one.ctypes.data, a.data.ptr, n, b.data.ptr, n)

        # ormqr (Computes Q * Z)
        ws_size = ormqr_helper(
            cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, no_trans, n, nrhs,
            mn_min, a.data.ptr, n, tau.data.ptr, b.data.ptr, n)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        ormqr(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, no_trans, n, nrhs,
              mn_min, a.data.ptr, n, tau.data.ptr, b.data.ptr, n,
              workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            ormqr, dev_info)

        return b


def _batched_posv(a, b):

    if not _cupy.cusolver.check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    dtype = _numpy.promote_types(a.dtype, b.dtype)
    dtype = _numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrfBatched = _cusolver.spotrfBatched
        potrsBatched = _cusolver.spotrsBatched
    elif dtype == 'd':
        potrfBatched = _cusolver.dpotrfBatched
        potrsBatched = _cusolver.dpotrsBatched
    elif dtype == 'F':
        potrfBatched = _cusolver.cpotrfBatched
        potrsBatched = _cusolver.cpotrsBatched
    elif dtype == 'D':
        potrfBatched = _cusolver.zpotrfBatched
        potrsBatched = _cusolver.zpotrsBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    a = a.astype(dtype, order='C', copy=True)
    ap = _cupy._core._mat_ptrs(a)
    lda, n = a.shape[-2:]
    batch_size = int(_numpy.prod(a.shape[:-2]))

    handle = _device.get_cusolver_handle()
    uplo = _cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = _cupy.empty(batch_size, dtype=_numpy.int32)

    # Cholesky factorization
    potrfBatched(handle, uplo, n, ap.data.ptr, lda, dev_info.data.ptr,
                 batch_size)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

    b_shape = b.shape
    b = b.conj().reshape(batch_size, n, -1).astype(dtype, order='C', copy=True)
    bp = _cupy._core._mat_ptrs(b)
    ldb, nrhs = b.shape[-2:]
    dev_info = _cupy.empty(1, dtype=_numpy.int32)

    # NOTE: potrsBatched does not currently support nrhs > 1 (CUDA v10.2)
    # Solve: A[i] * X[i] = B[i]
    potrsBatched(handle, uplo, n, nrhs, ap.data.ptr, lda, bp.data.ptr, ldb,
                 dev_info.data.ptr, batch_size)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrsBatched, dev_info)

    # TODO: check if conj() is necessary when nrhs > 1
    return b.conj().reshape(b_shape)


def posv(a, b):
    """Solve the linear equations A x = b via Cholesky factorization of A,
    where A is a real symmetric or complex Hermitian positive-definite matrix.

    If matrix ``A`` is not positive definite, Cholesky factorization fails
    and it raises an error.

    Note: For batch input, NRHS > 1 is not currently supported.

    Args:
        a (cupy.ndarray): Array of real symmetric or complex hermitian
            matrices with dimension (..., N, N).
        b (cupy.ndarray): right-hand side (..., N) or (..., N, NRHS).
    Returns:
        x (cupy.ndarray): The solution (shape matches b).
    """

    _cupy.linalg._util._assert_cupy_array(a, b)
    _cupy.linalg._util._assert_nd_squareness(a)

    if a.ndim > 2:
        return _batched_posv(a, b)

    dtype = _numpy.promote_types(a.dtype, b.dtype)
    dtype = _numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrf = _cusolver.spotrf
        potrf_bufferSize = _cusolver.spotrf_bufferSize
        potrs = _cusolver.spotrs
    elif dtype == 'd':
        potrf = _cusolver.dpotrf
        potrf_bufferSize = _cusolver.dpotrf_bufferSize
        potrs = _cusolver.dpotrs
    elif dtype == 'F':
        potrf = _cusolver.cpotrf
        potrf_bufferSize = _cusolver.cpotrf_bufferSize
        potrs = _cusolver.cpotrs
    elif dtype == 'D':
        potrf = _cusolver.zpotrf
        potrf_bufferSize = _cusolver.zpotrf_bufferSize
        potrs = _cusolver.zpotrs
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    a = a.astype(dtype, order='F', copy=True)
    lda, n = a.shape

    handle = _device.get_cusolver_handle()
    uplo = _cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = _cupy.empty(1, dtype=_numpy.int32)

    worksize = potrf_bufferSize(handle, uplo, n, a.data.ptr, lda)
    workspace = _cupy.empty(worksize, dtype=dtype)

    # Cholesky factorization
    potrf(handle, uplo, n, a.data.ptr, lda, workspace.data.ptr,
          worksize, dev_info.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    b_shape = b.shape
    b = b.reshape(n, -1).astype(dtype, order='F', copy=True)
    ldb, nrhs = b.shape

    # Solve: A * X = B
    potrs(handle, uplo, n, nrhs, a.data.ptr, lda, b.data.ptr, ldb,
          dev_info.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrs, dev_info)

    return _cupy.ascontiguousarray(b.reshape(b_shape))
