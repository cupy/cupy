import numpy

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.linalg import util


def cholesky(a):
    """Cholesky decomposition.

    Decompose a given two-dimensional square matrix into ``L * L.T``,
    where ``L`` is a lower-triangular matrix and ``.T`` is a conjugate
    transpose operator.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(N, N)``

    Returns:
        cupy.ndarray: The lower-triangular matrix.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.cholesky`
    """
    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f').char

    x = a.astype(dtype, order='C', copy=True)
    n = len(a)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
    elif dtype == 'd':
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    elif dtype == 'F':
        potrf = cusolver.cpotrf
        potrf_bufferSize = cusolver.cpotrf_bufferSize
    else:  # dtype == 'D':
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize

    buffersize = potrf_bufferSize(
        handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=dtype)
    potrf(
        handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

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

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.qr`
    """
    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a)
    util._assert_rank2(a)

    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full', 'e', 'economic'):
            msg = 'The deprecated mode \'{}\' is not supported'.format(mode)
            raise ValueError(msg)
        else:
            raise ValueError('Unrecognized mode \'{}\''.format(mode))

    # support float32, float64, complex64, and complex128
    if a.dtype.char in 'fdFD':
        dtype = a.dtype.char
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f').char

    m, n = a.shape
    mn = min(m, n)
    if mn == 0:
        if mode == 'reduced':
            return cupy.empty((m, 0), dtype), cupy.empty((0, n), dtype)
        elif mode == 'complete':
            return cupy.identity(m, dtype), cupy.empty((m, n), dtype)
        elif mode == 'r':
            return cupy.empty((0, n), dtype)
        else:  # mode == 'raw'
            # compatibility with numpy.linalg.qr
            dtype = numpy.promote_types(dtype, 'd')
            return cupy.empty((n, m), dtype), cupy.empty((0,), dtype)

    x = a.transpose().astype(dtype, order='C', copy=True)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        geqrf_bufferSize = cusolver.sgeqrf_bufferSize
        geqrf = cusolver.sgeqrf
    elif dtype == 'd':
        geqrf_bufferSize = cusolver.dgeqrf_bufferSize
        geqrf = cusolver.dgeqrf
    elif dtype == 'F':
        geqrf_bufferSize = cusolver.cgeqrf_bufferSize
        geqrf = cusolver.cgeqrf
    elif dtype == 'D':
        geqrf_bufferSize = cusolver.zgeqrf_bufferSize
        geqrf = cusolver.zgeqrf
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    # compute working space of geqrf and solve R
    buffersize = geqrf_bufferSize(handle, m, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=dtype)
    tau = cupy.empty(mn, dtype=dtype)
    geqrf(handle, m, n, x.data.ptr, m,
          tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        geqrf, dev_info)

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
        elif a.dtype.char == 'F':
            # The same applies to complex64
            return x.astype(numpy.complex128), tau.astype(numpy.complex128)
        return x, tau

    if mode == 'complete' and m > n:
        mc = m
        q = cupy.empty((m, m), dtype)
    else:
        mc = mn
        q = cupy.empty((n, m), dtype)
    q[:n] = x

    # compute working space of orgqr and solve Q
    if dtype == 'f':
        orgqr_bufferSize = cusolver.sorgqr_bufferSize
        orgqr = cusolver.sorgqr
    elif dtype == 'd':
        orgqr_bufferSize = cusolver.dorgqr_bufferSize
        orgqr = cusolver.dorgqr
    elif dtype == 'F':
        orgqr_bufferSize = cusolver.cungqr_bufferSize
        orgqr = cusolver.cungqr
    elif dtype == 'D':
        orgqr_bufferSize = cusolver.zungqr_bufferSize
        orgqr = cusolver.zungqr

    buffersize = orgqr_bufferSize(
        handle, m, mc, mn, q.data.ptr, m, tau.data.ptr)
    workspace = cupy.empty(buffersize, dtype=dtype)
    orgqr(
        handle, m, mc, mn, q.data.ptr, m, tau.data.ptr, workspace.data.ptr,
        buffersize, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        orgqr, dev_info)

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

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.svd`
    """
    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a)
    util._assert_rank2(a)

    # Cast to float32 or float64
    a_dtype = numpy.promote_types(a.dtype.char, 'f').char
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
        gesvd = cusolver.sgesvd
        gesvd_bufferSize = cusolver.sgesvd_bufferSize
    elif a_dtype == 'd':
        gesvd = cusolver.dgesvd
        gesvd_bufferSize = cusolver.dgesvd_bufferSize
    elif a_dtype == 'F':
        gesvd = cusolver.cgesvd
        gesvd_bufferSize = cusolver.cgesvd_bufferSize
    else:  # a_dtype == 'D':
        gesvd = cusolver.zgesvd
        gesvd_bufferSize = cusolver.zgesvd_bufferSize

    buffersize = gesvd_bufferSize(handle, m, n)
    workspace = cupy.empty(buffersize, dtype=a_dtype)
    gesvd(
        handle, job, job, m, n, x.data.ptr, m, s.data.ptr, u_ptr, m, vt_ptr, n,
        workspace.data.ptr, buffersize, 0, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        gesvd, dev_info)

    # Note that the returned array may need to be transporsed
    # depending on the structure of an input
    if compute_uv:
        if trans_flag:
            return u.transpose(), s, vt.transpose()
        else:
            return vt, s, u
    else:
        return s
