import numpy
from numpy import linalg

import cupy
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import util

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def cholesky(a):
    """Cholesky decomposition.

    Decompose a given two-dimensional square matrix into ``L * L.T``,
    where ``L`` is a lower-triangular matrix and ``.T`` is a conjugate
    transpose operator. Note that in the current implementation ``a`` must be
    a real matrix, and only float32 and float64 are supported.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(N, N)``

    Returns:
        cupy.ndarray: The lower-triangular matrix.

    .. seealso:: :func:`numpy.linalg.cholesky`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    x = a.astype(dtype, order='C', copy=True)
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
    util._tril(x, k=0)
    return x


def qr(a, mode='reduced'):
    """QR decomposition.

    Decompose a given two-dimensional matrix into ``Q * R``, where ``Q``
    is an orthonormal and ``R`` is an upper-triangular matrix.

    Args:
        a (cupy.ndarray): The input matrix.
        mode (str): The mode of decomposition. Currently 'reduced',
            'complete', 'r', and 'raw' modes are supported. The default mode
            is 'reduced', in which matrix ``A = (M, N)`` is decomposed into
            ``Q``, ``R`` with dimensions ``(M, K)``, ``(K, N)``, where
            ``K = min(M, N)``.

    Returns:
        cupy.ndarray, or tuple of ndarray:
            Although the type of returned object depends on the mode,
            it returns a tuple of ``(Q, R)`` by default.
            For details, please see the document of :func:`numpy.linalg.qr`.

    .. seealso:: :func:`numpy.linalg.qr`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a)
    util._assert_rank2(a)

    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full', 'e', 'economic'):
            msg = 'The deprecated mode \'{}\' is not supported'.format(mode)
            raise ValueError(msg)
        else:
            raise ValueError('Unrecognized mode \'{}\''.format(mode))

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    m, n = a.shape
    x = a.transpose().astype(dtype, order='C', copy=True)
    mn = min(m, n)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    # compute working space of geqrf and ormqr, and solve R
    if dtype == 'f':
        buffersize = cusolver.sgeqrf_bufferSize(handle, m, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        tau = cupy.empty(mn, dtype=numpy.float32)
        cusolver.sgeqrf(
            handle, m, n, x.data.ptr, m,
            tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)
    else:  # dtype == 'd'
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
        r = x[:, :mn].transpose()
        return util._triu(r)

    if mode == 'raw':
        if a.dtype.char == 'f':
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
    if dtype == 'f':
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

    q = q[:mc].transpose()
    r = x[:, :mc].transpose()
    return q, util._triu(r)


def svd(a, full_matrices=True, compute_uv=True):
    """Singular Value Decomposition.

    Factorizes the matrix ``a`` as ``u * np.diag(s) * v``, where ``u`` and
    ``v`` are unitary and ``s`` is an one-dimensional array of ``a``'s
    singular values.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(M, N)``.
        full_matrices (bool): If True, it returns u and v with dimensions
            ``(M, M)`` and ``(N, N)``. Otherwise, the dimensions of u and v
            are respectively ``(M, K)`` and ``(K, N)``, where
            ``K = min(M, N)``.
        compute_uv (bool): If True, it only returns singular values.

    Returns:
        tuple of :class:`cupy.ndarray`:
            A tuple of ``(u, s, v)`` such that ``a = u * np.diag(s) * v``.

    .. seealso:: :func:`numpy.linalg.svd`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a)
    util._assert_rank2(a)

    # Cast to float32 or float64
    a_dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char
    if a_dtype == 'f':
        s_dtype = 'f'
    elif a_dtype == 'd':
        s_dtype = 'd'
    elif a_dtype == 'F':
        s_dtype = 'f'
    else:  # a_dtype == 'D':
        a_dtype = 'D'
        s_dtype = 'd'

    # Remark 1: gesvd only supports m >= n (WHAT?)
    # Remark 2: gesvd only supports jobu = 'A' and jobvt = 'A'
    # Remark 3: gesvd returns matrix U and V^H
    # Remark 4: Remark 2 is removed since cuda 8.0 (new!)
    
    n, p = a.shape
    if p > n:
        trans_flag = True
        # Perform SVD(X.T) not SVD(X) This reduces complexity from
        # O(np^2) to O(n^2p) if p > n
        if a.flags.c_contiguous:
            X = a.astype(a_dtype, order = 'C', copy = True).transpose()
        else:
            X = a.transpose().astype(a_dtype, order = 'F', copy = True)
        n, p = X.shape
    else:
        trans_flag = False
        X = a.astype(a_dtype, order = 'F', copy = True)


    MIN = min(n, p)

    if compute_uv:
        if full_matrices:
            U = cupy.empty((n, n), dtype = a_dtype, order = 'F')
            VT = X
            jobU = ord('A')
            jobVT = ord('O')
            U_ptr = U.data.ptr
            VT_ptr = X.data.ptr
        else:
            U = X
            VT = cupy.empty((MIN, p), dtype = a_dtype, order = 'F')
            jobU = ord('O')
            jobVT = ord('S')
            U_ptr = X.data.ptr
            VT_ptr = VT.data.ptr
    else:
        U_ptr, VT_ptr = 0, 0  # Use nullptr
        jobU = ord('N')
        jobVT = ord('N')

        
    S = cupy.empty(MIN, dtype = s_dtype)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype = numpy.int32)

        
    if a_dtype == 'f':
        buffersize = cusolver.sgesvd_bufferSize(handle, n, p)
        workspace = cupy.empty(buffersize, dtype = a_dtype)
        cusolver.sgesvd(
            handle, jobU, jobVT, n, p, X.data.ptr, n,
            S.data.ptr, U_ptr, n, VT_ptr, p,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)

    elif a_dtype == 'd':
        buffersize = cusolver.dgesvd_bufferSize(handle, n, p)
        workspace = cupy.empty(buffersize, dtype = a_dtype)
        cusolver.dgesvd(
            handle, jobU, jobVT, n, p, X.data.ptr, n,
            S.data.ptr, U_ptr, n, VT_ptr, p,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)

    elif a_dtype == 'F':
        buffersize = cusolver.cgesvd_bufferSize(handle, n, p)
        workspace = cupy.empty(buffersize, dtype = a_dtype)
        cusolver.cgesvd(
            handle, jobU, jobVT, n, p, X.data.ptr, n,
            S.data.ptr, U_ptr, n, VT_ptr, p,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)

    else:  # a_dtype == 'D':
        buffersize = cusolver.zgesvd_bufferSize(handle, n, p)
        workspace = cupy.empty(buffersize, dtype = a_dtype)
        cusolver.zgesvd(
            handle, jobU, jobVT, n, p, X.data.ptr, n,
            S.data.ptr, U_ptr, n, VT_ptr, p,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)


    status = int(dev_info[0])
    if status > 0:
        raise linalg.LinAlgError(
            'SVD computation does not converge')
    elif status < 0:
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')

    # Note that the returned array may need to be transprosed
    # depending on the structure of an input
    if compute_uv:
        if trans_flag:
            return VT.transpose(), S, U.transpose()
        else:
            return U, S, VT
    else:
        return S
