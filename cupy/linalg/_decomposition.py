import numpy

import cupy
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy._core import internal
from cupy.cuda import device
from cupy.cusolver import check_availability
from cupy.cusolver import _gesvdj_batched, _gesvd_batched
from cupy.linalg import _util


def _lu_factor(a_t, dtype):
    """Compute pivoted LU decomposition.

    Decompose a given batch of square matrices. Inputs and outputs are
    transposed.

    Args:
        a_t (cupy.ndarray): The input matrix with dimension ``(..., N, N)``.
            The dimension condition is not checked.
        dtype (numpy.dtype): float32, float64, complex64, or complex128.

    Returns:
        lu_t (cupy.ndarray): ``L`` without its unit diagonal and ``U`` with
            dimension ``(..., N, N)``.
        piv (cupy.ndarray): 1-origin pivot indices with dimension
            ``(..., N)``.
        dev_info (cupy.ndarray): ``getrf`` info with dimension ``(...)``.

    .. seealso:: :func:`scipy.linalg.lu_factor`

    """
    orig_shape = a_t.shape
    n = orig_shape[-2]

    # copy is necessary to present `a` to be overwritten.
    a_t = a_t.astype(dtype, order='C').reshape(-1, n, n)
    batch_size = a_t.shape[0]
    ipiv = cupy.empty((batch_size, n), dtype=numpy.int32)
    dev_info = cupy.empty((batch_size,), dtype=numpy.int32)

    # Heuristic condition from some performance test.
    # TODO(kataoka): autotune
    use_batched = batch_size * 65536 >= n * n

    if use_batched:
        handle = device.get_cublas_handle()
        lda = n
        step = n * lda * a_t.itemsize
        start = a_t.data.ptr
        stop = start + step * batch_size
        a_array = cupy.arange(start, stop, step, dtype=cupy.uintp)

        if dtype == numpy.float32:
            getrfBatched = cupy.cuda.cublas.sgetrfBatched
        elif dtype == numpy.float64:
            getrfBatched = cupy.cuda.cublas.dgetrfBatched
        elif dtype == numpy.complex64:
            getrfBatched = cupy.cuda.cublas.cgetrfBatched
        elif dtype == numpy.complex128:
            getrfBatched = cupy.cuda.cublas.zgetrfBatched
        else:
            assert False

        getrfBatched(
            handle, n, a_array.data.ptr, lda, ipiv.data.ptr,
            dev_info.data.ptr, batch_size)

    else:
        handle = device.get_cusolver_handle()
        if dtype == numpy.float32:
            getrf_bufferSize = cusolver.sgetrf_bufferSize
            getrf = cusolver.sgetrf
        elif dtype == numpy.float64:
            getrf_bufferSize = cusolver.dgetrf_bufferSize
            getrf = cusolver.dgetrf
        elif dtype == numpy.complex64:
            getrf_bufferSize = cusolver.cgetrf_bufferSize
            getrf = cusolver.cgetrf
        elif dtype == numpy.complex128:
            getrf_bufferSize = cusolver.zgetrf_bufferSize
            getrf = cusolver.zgetrf
        else:
            assert False

        for i in range(batch_size):
            a_ptr = a_t[i].data.ptr
            buffersize = getrf_bufferSize(handle, n, n, a_ptr, n)
            workspace = cupy.empty(buffersize, dtype=dtype)
            getrf(
                handle, n, n, a_ptr, n, workspace.data.ptr,
                ipiv[i].data.ptr, dev_info[i].data.ptr)

    return (
        a_t.reshape(orig_shape),
        ipiv.reshape(orig_shape[:-1]),
        dev_info.reshape(orig_shape[:-2]),
    )


def _potrf_batched(a):
    """Batched Cholesky decomposition.

    Decompose a given array of two-dimensional square matrices into
    ``L * L.T``, where ``L`` is a lower-triangular matrix and ``.T``
    is a conjugate transpose operator.

    Args:
        a (cupy.ndarray): The input array of matrices
            with dimension ``(..., N, N)``

    Returns:
        cupy.ndarray: The lower-triangular matrix.
    """
    if not check_availability('potrfBatched'):
        raise RuntimeError('potrfBatched is not available')

    dtype, out_dtype = _util.linalg_common_type(a)

    if dtype == 'f':
        potrfBatched = cusolver.spotrfBatched
    elif dtype == 'd':
        potrfBatched = cusolver.dpotrfBatched
    elif dtype == 'F':
        potrfBatched = cusolver.cpotrfBatched
    else:  # dtype == 'D':
        potrfBatched = cusolver.zpotrfBatched

    x = a.astype(dtype, order='C', copy=True)
    xp = cupy._core._mat_ptrs(x)
    n = x.shape[-1]
    ldx = x.strides[-2] // x.dtype.itemsize
    handle = device.get_cusolver_handle()
    batch_size = internal.prod(x.shape[:-2])
    dev_info = cupy.empty(batch_size, dtype=numpy.int32)

    potrfBatched(
        handle, cublas.CUBLAS_FILL_MODE_UPPER, n, xp.data.ptr, ldx,
        dev_info.data.ptr, batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

    return cupy.tril(x).astype(out_dtype, copy=False)


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
    _util._assert_cupy_array(a)
    _util._assert_nd_squareness(a)

    if a.ndim > 2:
        return _potrf_batched(a)

    dtype, out_dtype = _util.linalg_common_type(a)

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
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    _util._tril(x, k=0)
    return x.astype(out_dtype, copy=False)


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
    _util._assert_cupy_array(a)
    _util._assert_rank2(a)

    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full', 'e', 'economic'):
            msg = 'The deprecated mode \'{}\' is not supported'.format(mode)
            raise ValueError(msg)
        else:
            raise ValueError('Unrecognized mode \'{}\''.format(mode))

    # support float32, float64, complex64, and complex128
    dtype, out_dtype = _util.linalg_common_type(a)
    if mode == 'raw':
        # compatibility with numpy.linalg.qr
        out_dtype = numpy.promote_types(out_dtype, 'd')

    m, n = a.shape
    mn = min(m, n)
    if mn == 0:
        if mode == 'reduced':
            return cupy.empty((m, 0), out_dtype), cupy.empty((0, n), out_dtype)
        elif mode == 'complete':
            return cupy.identity(m, out_dtype), cupy.empty((m, n), out_dtype)
        elif mode == 'r':
            return cupy.empty((0, n), out_dtype)
        else:  # mode == 'raw'
            return cupy.empty((n, m), out_dtype), cupy.empty((0,), out_dtype)

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
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        geqrf, dev_info)

    if mode == 'r':
        r = x[:, :mn].transpose()
        return _util._triu(r).astype(out_dtype, copy=False)

    if mode == 'raw':
        return (
            x.astype(out_dtype, copy=False),
            tau.astype(out_dtype, copy=False))

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
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        orgqr, dev_info)

    q = q[:mc].transpose()
    r = x[:, :mc].transpose()
    return (
        q.astype(out_dtype, copy=False),
        _util._triu(r).astype(out_dtype, copy=False))


def _svd_batched(a, full_matrices, compute_uv):
    batch_shape = a.shape[:-2]
    batch_size = internal.prod(batch_shape)
    n, m = a.shape[-2:]

    dtype, uv_dtype = _util.linalg_common_type(a)
    s_dtype = uv_dtype.char.lower()

    # first handle any 0-size inputs
    if batch_size == 0:
        k = min(m, n)
        s = cupy.empty(batch_shape + (k,), s_dtype)
        if compute_uv:
            if full_matrices:
                u = cupy.empty(batch_shape + (n, n), dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (m, m), dtype=uv_dtype)
            else:
                u = cupy.empty(batch_shape + (n, k), dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (k, m), dtype=uv_dtype)
            return u, s, vt
        else:
            return s
    elif m == 0 or n == 0:
        s = cupy.empty(batch_shape + (0,), s_dtype)
        if compute_uv:
            if full_matrices:
                u = cupy.empty(batch_shape + (n, n), dtype=uv_dtype)
                u[...] = cupy.identity(n, dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (m, m), dtype=uv_dtype)
                vt[...] = cupy.identity(m, dtype=uv_dtype)
            else:
                u = cupy.empty(batch_shape + (n, 0), dtype=uv_dtype)
                vt = cupy.empty(batch_shape + (0, m), dtype=uv_dtype)
            return u, s, vt
        else:
            return s

    # ...then delegate real computation to cuSOLVER
    a = a.reshape(-1, *(a.shape[-2:]))
    if runtime.is_hip or (m <= 32 and n <= 32):
        # copy is done in _gesvdj_batched, so let's try not to do it here
        a = a.astype(dtype, order='C', copy=False)
        out = _gesvdj_batched(a, full_matrices, compute_uv, False)
    else:
        # manually loop over cusolverDn<t>gesvd()
        # copy (via possible type casting) is done in _gesvd_batched
        # note: _gesvd_batched returns V, not V^H
        out = _gesvd_batched(a, dtype.char, full_matrices, compute_uv, False)
    if compute_uv:
        u, s, v = out
        u = u.astype(uv_dtype, copy=False)
        u = u.reshape(*batch_shape, *(u.shape[-2:]))
        s = s.astype(s_dtype, copy=False)
        s = s.reshape(*batch_shape, *(s.shape[-1:]))
        v = v.astype(uv_dtype, copy=False)
        v = v.reshape(*batch_shape, *(v.shape[-2:]))
        return u, s, v.swapaxes(-2, -1).conj()
    else:
        s = out
        s = s.astype(s_dtype, copy=False)
        s = s.reshape(*batch_shape, *(s.shape[-1:]))
        return s


# TODO(leofang): support the hermitian keyword?
def svd(a, full_matrices=True, compute_uv=True):
    """Singular Value Decomposition.

    Factorizes the matrix ``a`` as ``u * np.diag(s) * v``, where ``u`` and
    ``v`` are unitary and ``s`` is an one-dimensional array of ``a``'s
    singular values.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(..., M, N)``.
        full_matrices (bool): If True, it returns u and v with dimensions
            ``(..., M, M)`` and ``(..., N, N)``. Otherwise, the dimensions
            of u and v are ``(..., M, K)`` and ``(..., K, N)``, respectively,
            where ``K = min(M, N)``.
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

    .. note::
        On CUDA, when ``a.ndim > 2`` and the matrix dimensions <= 32, a fast
        code path based on Jacobian method (``gesvdj``) is taken. Otherwise,
        a QR method (``gesvd``) is used.

        On ROCm, there is no such a fast code path that switches the underlying
        algorithm.

    .. seealso:: :func:`numpy.linalg.svd`
    """
    _util._assert_cupy_array(a)
    if a.ndim > 2:
        return _svd_batched(a, full_matrices, compute_uv)

    # Cast to float32 or float64
    dtype, uv_dtype = _util.linalg_common_type(a)
    real_dtype = dtype.char.lower()
    s_dtype = uv_dtype.char.lower()

    # Remark 1: gesvd only supports m >= n (WHAT?)
    # Remark 2: gesvd returns matrix U and V^H
    n, m = a.shape

    if m == 0 or n == 0:
        s = cupy.empty((0,), s_dtype)
        if compute_uv:
            if full_matrices:
                u = cupy.eye(n, dtype=uv_dtype)
                vt = cupy.eye(m, dtype=uv_dtype)
            else:
                u = cupy.empty((n, 0), dtype=uv_dtype)
                vt = cupy.empty((0, m), dtype=uv_dtype)
            return u, s, vt
        else:
            return s

    # `a` must be copied because xgesvd destroys the matrix
    if m >= n:
        x = a.astype(dtype, order='C', copy=True)
        trans_flag = False
    else:
        m, n = a.shape
        x = a.transpose().astype(dtype, order='C', copy=True)
        trans_flag = True

    k = n  # = min(m, n) where m >= n is ensured above
    if compute_uv:
        if full_matrices:
            u = cupy.empty((m, m), dtype=dtype)
            vt = x[:, :n]
            job_u = ord('A')
            job_vt = ord('O')
        else:
            u = x
            vt = cupy.empty((k, n), dtype=dtype)
            job_u = ord('O')
            job_vt = ord('S')
        u_ptr, vt_ptr = u.data.ptr, vt.data.ptr
    else:
        u_ptr, vt_ptr = 0, 0  # Use nullptr
        job_u = ord('N')
        job_vt = ord('N')
    s = cupy.empty(k, dtype=real_dtype)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        gesvd = cusolver.sgesvd
        gesvd_bufferSize = cusolver.sgesvd_bufferSize
    elif dtype == 'd':
        gesvd = cusolver.dgesvd
        gesvd_bufferSize = cusolver.dgesvd_bufferSize
    elif dtype == 'F':
        gesvd = cusolver.cgesvd
        gesvd_bufferSize = cusolver.cgesvd_bufferSize
    else:  # dtype == 'D':
        gesvd = cusolver.zgesvd
        gesvd_bufferSize = cusolver.zgesvd_bufferSize

    buffersize = gesvd_bufferSize(handle, m, n)
    workspace = cupy.empty(buffersize, dtype=dtype)
    if not runtime.is_hip:
        # rwork can be NULL if the information from supperdiagonal isn't needed
        # https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-gesvd  # noqa
        rwork_ptr = 0
    else:
        rwork = cupy.empty(min(m, n)-1, dtype=s_dtype)
        rwork_ptr = rwork.data.ptr
    gesvd(
        handle, job_u, job_vt, m, n, x.data.ptr, m, s.data.ptr, u_ptr, m,
        vt_ptr, n, workspace.data.ptr, buffersize, rwork_ptr,
        dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        gesvd, dev_info)

    s = s.astype(s_dtype, copy=False)

    # Note that the returned array may need to be transposed
    # depending on the structure of an input
    if compute_uv:
        u = u.astype(uv_dtype, copy=False)
        vt = vt.astype(uv_dtype, copy=False)
        if trans_flag:
            return u.transpose(), s, vt.transpose()
        else:
            return vt, s, u
    else:
        return s
