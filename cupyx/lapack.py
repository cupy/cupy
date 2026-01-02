from __future__ import annotations

import numpy as _numpy

import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
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
    from cupy_backends.cuda.libs import cusolver as _cusolver

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
    from cupy_backends.cuda.libs import cusolver as _cusolver

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
    from cupy_backends.cuda.libs import cusolver as _cusolver
    import cupyx.cusolver

    if not cupyx.cusolver.check_availability('potrsBatched'):
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
    from cupy_backends.cuda.libs import cusolver as _cusolver

    _util = _cupy.linalg._util
    _util._assert_cupy_array(a, b)
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

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


def potrs(L, b, lower):
    """ Implements lapack XPOTRS through cusolver.potrs. Solves linear system
    A x = b given the cholesky decomposition of A, namely L. Supports also
    batches of linear systems and more than one right-hand side (NRHS > 1).

    Args:
        L (cupy.ndarray): Array of Cholesky decomposition of real symmetric or
            complex hermitian matrices with dimension (..., N, N).
        b (cupy.ndarray): right-hand side (..., N) or (..., N, NRHS). Note that
            this array may be modified in place, as usually done in LAPACK.
        lower (bool): If True, L is lower triangular. If False, L is upper
            triangular.

    Returns:
        cupy.ndarray: The solution to the linear system. Note this may point to
            the same memory as b, since b may be modified in place.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    """

    from cupy_backends.cuda.libs import cusolver as _cusolver

    _util = _cupy.linalg._util
    _util._assert_cupy_array(L, b)
    _util._assert_stacked_2d(L)
    _util._assert_stacked_square(L)

    # Check if batched should be used
    if L.ndim > 2:
        return _batched_potrs(L, b, lower)

    # Check input arguments
    n = L.shape[-1]
    b_shape = b.shape
    if b.ndim == 1:
        b = b[:, None]
    assert b.ndim == 2, "b is not a vector or a matrix"
    assert b.shape[0] == n, "length of arrays in b does not match size of L"

    # Check memory order and type
    dtype = _numpy.promote_types(L.dtype, b.dtype)
    dtype = _numpy.promote_types(dtype, 'f')
    L, b = L.astype(dtype, copy=False), b.astype(dtype, copy=False)
    if (not L.flags.f_contiguous) and (not L.flags.c_contiguous):
        L = _cupy.asfortranarray(L)
    if L.flags.c_contiguous:
        lower = not lower  # Cusolver assumes F-order
        # For complex types, we need to conjugate the matrix
        if b.size < L.size:  # Conjugate the one with lower memory footprint
            b = b.conj()
        else:
            L = L.conj()
    if (b.dtype != dtype) or (not b.flags.f_contiguous):
        b = _cupy.asfortranarray(b)

    # Take correct dtype
    if dtype == 'f':
        potrs = _cusolver.spotrs
    elif dtype == 'd':
        potrs = _cusolver.dpotrs
    elif dtype == 'F':
        potrs = _cusolver.cpotrs
    elif dtype == 'D':
        potrs = _cusolver.zpotrs
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(L.dtype))
        raise ValueError(msg)

    handle = _device.get_cusolver_handle()
    dev_info = _cupy.empty(1, dtype=_cupy.int32)

    potrs(
        handle,
        _cublas.CUBLAS_FILL_MODE_LOWER if lower else
        _cublas.CUBLAS_FILL_MODE_UPPER,
        L.shape[0],  # n, matrix size
        b.shape[1],  # nrhs
        L.data.ptr,
        L.shape[0],  # ldL
        b.data.ptr,
        b.shape[0],  # ldB
        dev_info.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrs, dev_info)

    # Conjugate back if necessary
    if L.flags.c_contiguous and b.size < L.size:
        b = b.conj()
    return b.reshape(b_shape)


def _batched_potrs(L, b, lower: bool):
    from cupy_backends.cuda.libs import cusolver as _cusolver
    import cupyx.cusolver

    if not cupyx.cusolver.check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    # CHeck input arrays
    assert b.ndim >= L.ndim-1, "Batch dimension of b is different than that \
        of L"
    b_shape = b.shape
    if b.ndim < L.ndim:
        b = b[..., None]
    assert b.shape[:-2] == L.shape[:-2], \
        "Batch dimension of L and b do not match"

    # Check dtype and memory alignment
    dtype = _numpy.promote_types(L.dtype, b.dtype)
    dtype = _numpy.promote_types(dtype, 'f')
    L = L.astype(dtype, order='C', copy=False)
    b = b.astype(dtype, order='C', copy=False)
    assert L.flags.c_contiguous and b.flags.c_contiguous, \
        "Unexpected non C-contiguous arrays"
    lower = not lower  # Cusolver assumes F-order

    # Pick function handle
    if dtype == 'f':
        potrsBatched = _cusolver.spotrsBatched
    elif dtype == 'd':
        potrsBatched = _cusolver.dpotrsBatched
    elif dtype == 'F':
        potrsBatched = _cusolver.cpotrsBatched
    elif dtype == 'D':
        potrsBatched = _cusolver.zpotrsBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(L.dtype))
        raise ValueError(msg)

    # Variables for potrs batched
    handle = _device.get_cusolver_handle()
    dev_info = _cupy.empty(1, dtype=_numpy.int32)
    batch_size = _numpy.prod(L.shape[:-2])
    n = L.shape[-1]
    b = b.conj()
    L_p = _cupy._core._mat_ptrs(L)
    nrhs = b.shape[-1]

    # Allocate temporary working array in case nrhs > 1
    if nrhs == 1:
        b_tmp = b[..., 0]
    else:
        b_tmp = _cupy.empty(b.shape[:-1], dtype=b.dtype, order='C')
    b_tmp_p = _cupy._core._mat_ptrs(b_tmp[..., None])

    # potrs_batched supports only nrhs=1, so we have to loop over the nrhs
    for i in range(b.shape[-1]):

        if nrhs > 1:  # Copy results back to the original array
            b_tmp[...] = b[..., i]

        potrsBatched(
            handle,
            _cublas.CUBLAS_FILL_MODE_LOWER if lower else
            _cublas.CUBLAS_FILL_MODE_UPPER,
            n,  # n
            1,  # nrhs
            L_p.data.ptr,  # A
            L.shape[-2],  # lda
            b_tmp_p.data.ptr,  # Barray
            b_tmp.shape[-1],  # ldb
            dev_info.data.ptr,  # info
            batch_size  # batchSize
        )
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
            potrsBatched, dev_info)

        if nrhs > 1:  # Copy results back to the original array
            b[..., i] = b_tmp.conj()
        else:
            b = b_tmp.conj()

    # Return b in the original shape
    return b.reshape(b_shape)
