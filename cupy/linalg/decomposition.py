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
        compute_uv (bool): If ``False``, it only returns singular values.

    Returns:
        tuple of :class:`cupy.ndarray`:
            A tuple of ``(u, s, v)`` such that ``a = u * np.diag(s) * v``.

    .. seealso:: :func:`numpy.linalg.svd`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    if a.ndim >= 3:
        return _batched_svd(a, full_matrices, compute_uv)

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
    n, m = a.shape

    # `a` must be copied because xgesvd destroys the matrix
    if m >= n:
        x = a.astype(a_dtype, order='C', copy=True)
        trans_flag = False
    else:
        m, n = a.shape
        x = a.transpose().astype(a_dtype, order='C', copy=True)
        trans_flag = True
    mn = min(m, n)

    if compute_uv:
        if full_matrices:
            u = cupy.empty((m, m), dtype=a_dtype)
            vt = cupy.empty((n, n), dtype=a_dtype)
        else:
            u = cupy.empty((mn, m), dtype=a_dtype)
            vt = cupy.empty((mn, n), dtype=a_dtype)
        u_ptr, vt_ptr = u.data.ptr, vt.data.ptr
    else:
        u_ptr, vt_ptr = 0, 0  # Use nullptr
    s = cupy.empty(mn, dtype=s_dtype)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    if compute_uv:
        job = ord('A') if full_matrices else ord('S')
    else:
        job = ord('N')
    if a_dtype == 'f':
        buffersize = cusolver.sgesvd_bufferSize(handle, m, n)
        workspace = cupy.empty(buffersize, dtype=a_dtype)
        cusolver.sgesvd(
            handle, job, job, m, n, x.data.ptr, m,
            s.data.ptr, u_ptr, m, vt_ptr, n,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)
    elif a_dtype == 'd':
        buffersize = cusolver.dgesvd_bufferSize(handle, m, n)
        workspace = cupy.empty(buffersize, dtype=a_dtype)
        cusolver.dgesvd(
            handle, job, job, m, n, x.data.ptr, m,
            s.data.ptr, u_ptr, m, vt_ptr, n,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)
    elif a_dtype == 'F':
        buffersize = cusolver.cgesvd_bufferSize(handle, m, n)
        workspace = cupy.empty(buffersize, dtype=a_dtype)
        cusolver.cgesvd(
            handle, job, job, m, n, x.data.ptr, m,
            s.data.ptr, u_ptr, m, vt_ptr, n,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)
    else:  # a_dtype == 'D':
        buffersize = cusolver.zgesvd_bufferSize(handle, m, n)
        workspace = cupy.empty(buffersize, dtype=a_dtype)
        cusolver.zgesvd(
            handle, job, job, m, n, x.data.ptr, m,
            s.data.ptr, u_ptr, m, vt_ptr, n,
            workspace.data.ptr, buffersize, 0, dev_info.data.ptr)

    status = int(dev_info[0])
    if status > 0:
        raise linalg.LinAlgError(
            'SVD computation does not converge')
    elif status < 0:
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')

    # Note that the returned array may need to be transporsed
    # depending on the structure of an input
    if compute_uv:
        if trans_flag:
            return u.transpose(), s, vt.transpose()
        else:
            return vt, s, u
    else:
        return s

def _batched_svd(a, full_matrices, compute_uv):
    util._assert_cupy_array(a)

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

    # `a` must be copied because xgesvd destroys the matrix
    a = a.astype(a_dtype, order='C', copy=True)

    assert a.size != 0  # TODO(kataoka): later
    a_shape = a.shape
    batch_shape = a_shape[:-2]
    n, m = a_shape[-2:]
    batch_size = cupy.core.internal.prod(batch_shape)

    mn = min(m, n)

    # TODO(kataoka): full_matrices=False case is not efficient
    if compute_uv:
        u = cupy.empty(batch_shape + (m, m), dtype=a_dtype)
        v = cupy.empty(batch_shape + (n, n), dtype=a_dtype)
        u_ptr, v_ptr = u.data.ptr, v.data.ptr
    else:
        # Somehow nullptr cannot be used
        tmp = cupy.empty(1, dtype=a_dtype)
        u_ptr, v_ptr = tmp.data.ptr, tmp.data.ptr
    s = cupy.empty(batch_shape + (mn,), dtype=s_dtype)
    handle = device.get_cusolver_handle()
    if compute_uv:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
    lda = m
    ldu = m
    ldv = n
    info = cupy.empty(batch_size, dtype=numpy.int32)
    if a_dtype == 'f':
        gesvdjBatched_bufferSize = cusolver.sgesvdjBatched_bufferSize
        gesvdjBatched = cusolver.sgesvdjBatched
    elif a_dtype == 'd':
        gesvdjBatched_bufferSize = cusolver.dgesvdjBatched_bufferSize
        gesvdjBatched = cusolver.dgesvdjBatched
    elif a_dtype == 'F':
        gesvdjBatched_bufferSize = cusolver.cgesvdjBatched_bufferSize
        gesvdjBatched = cusolver.cgesvdjBatched
    elif a_dtype == 'D':
        gesvdjBatched_bufferSize = cusolver.zgesvdjBatched_bufferSize
        gesvdjBatched = cusolver.zgesvdjBatched
    else:
        assert False

    buffersize = numpy.empty(1, numpy.int32)
    params = cusolver.createGesvdjInfo()
    buffersize = gesvdjBatched_bufferSize(
        handle, jobz, m, n, a.data.ptr, lda, s.data.ptr,
        u_ptr, ldu, v_ptr, ldv, params, batch_size)
    workspace = cupy.empty(buffersize, dtype=a_dtype)
    gesvdjBatched(
        handle, jobz, m, n, a.data.ptr, lda, s.data.ptr,
        u_ptr, ldu, v_ptr, ldv, workspace.data.ptr, buffersize,
        info.data.ptr, params, batch_size)
    cusolver.destroyGesvdjInfo(params)

    if (info < 0).any():
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')
    elif (info > 0).any():
        raise linalg.LinAlgError(
            'SVD computation does not converge')

    if compute_uv:
        if not full_matrices:
            u = u[..., :mn, :]
            v = v[..., :mn, :]
        # Note that the returned array may need to be transporsed
        # depending on the structure of an input
        return cupy.swapaxes(v, -1, -2), s, u
    else:
        return s
