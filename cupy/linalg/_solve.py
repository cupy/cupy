import warnings

import numpy
from numpy import linalg

import cupy
from cupy._core import internal
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import _decomposition
from cupy.linalg import _util
from cupy.cublas import batched_gesv, get_batched_gesv_limit
import cupyx


def solve(a, b):
    """Solves a linear matrix equation.

    It computes the exact solution of ``x`` in ``ax = b``,
    where ``a`` is a square and full rank matrix.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(..., M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(...,M)`` or
            ``(..., M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(..., M)`` or ``(..., M, K)``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.solve`
    """
    if a.ndim > 2 and a.shape[-1] <= get_batched_gesv_limit():
        # Note: There is a low performance issue in batched_gesv when matrix is
        # large, so it is not used in such cases.
        return batched_gesv(a, b)

    _util._assert_cupy_array(a, b)
    _util._assert_nd_squareness(a)

    if not ((a.ndim == b.ndim or a.ndim == b.ndim + 1) and
            a.shape[:-1] == b.shape[:a.ndim - 1]):
        raise ValueError(
            'a must have (..., M, M) shape and b must have (..., M) '
            'or (..., M, K)')

    dtype, out_dtype = _util.linalg_common_type(a, b)
    if a.ndim == 2:
        # prevent 'a' and 'b' to be overwritten
        a = a.astype(dtype, copy=True, order='F')
        b = b.astype(dtype, copy=True, order='F')
        cupyx.lapack.gesv(a, b)
        return b.astype(out_dtype, copy=False)

    # prevent 'a' to be overwritten
    a = a.astype(dtype, copy=True, order='C')
    x = cupy.empty_like(b, dtype=out_dtype)
    shape = a.shape[:-2]
    for i in range(numpy.prod(shape)):
        index = numpy.unravel_index(i, shape)
        # prevent 'b' to be overwritten
        bi = b[index].astype(dtype, copy=True, order='F')
        cupyx.lapack.gesv(a[index], bi)
        x[index] = bi
    return x


def _solve(a, b, cublas_handle, cusolver_handle):
    a = cupy.asfortranarray(a)
    b = cupy.asfortranarray(b)
    dtype = a.dtype
    m, k = (b.size, 1) if b.ndim == 1 else b.shape
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        geqrf = cusolver.sgeqrf
        geqrf_bufferSize = cusolver.sgeqrf_bufferSize
        ormqr = cusolver.sormqr
        ormqr_bufferSize = cusolver.sormqr_bufferSize
        trans = cublas.CUBLAS_OP_T
        trsm = cublas.strsm
    elif dtype == 'd':
        geqrf = cusolver.dgeqrf
        geqrf_bufferSize = cusolver.dgeqrf_bufferSize
        ormqr = cusolver.dormqr
        ormqr_bufferSize = cusolver.dormqr_bufferSize
        trans = cublas.CUBLAS_OP_T
        trsm = cublas.dtrsm
    elif dtype == 'F':
        geqrf = cusolver.cgeqrf
        geqrf_bufferSize = cusolver.cgeqrf_bufferSize
        ormqr = cusolver.cormqr
        ormqr_bufferSize = cusolver.cunmqr_bufferSize
        trans = cublas.CUBLAS_OP_C
        trsm = cublas.ctrsm
    elif dtype == 'D':
        geqrf = cusolver.zgeqrf
        geqrf_bufferSize = cusolver.zgeqrf_bufferSize
        ormqr = cusolver.zormqr
        ormqr_bufferSize = cusolver.zunmqr_bufferSize
        trans = cublas.CUBLAS_OP_C
        trsm = cublas.ztrsm
    else:
        raise NotImplementedError(dtype)

    # 1. QR decomposition (A = Q * R)
    buffersize = geqrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)
    tau = cupy.empty(m, dtype=dtype)
    geqrf(
        cusolver_handle, m, m, a.data.ptr, m, tau.data.ptr, workspace.data.ptr,
        buffersize, dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        geqrf, dev_info)
    # Explicitly free the space allocated by geqrf
    del workspace
    # 2. ormqr (Q^T * B)
    buffersize = ormqr_bufferSize(
        cusolver_handle, cublas.CUBLAS_SIDE_LEFT, trans, m, k, m, a.data.ptr,
        m, tau.data.ptr, b.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)
    ormqr(
        cusolver_handle, cublas.CUBLAS_SIDE_LEFT, trans, m, k, m, a.data.ptr,
        m, tau.data.ptr, b.data.ptr, m, workspace.data.ptr, buffersize,
        dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        ormqr, dev_info)

    # Explicitly free the space allocated by ormqr
    del workspace
    # 3. trsm (X = R^{-1} * (Q^T * B))
    one = numpy.array(1, dtype=dtype)
    trsm(
        cublas_handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
        cublas.CUBLAS_OP_N, cublas.CUBLAS_DIAG_NON_UNIT,
        m, k, one.ctypes.data, a.data.ptr, m, b.data.ptr, m)
    return b


def tensorsolve(a, b, axes=None):
    """Solves tensor equations denoted by ``ax = b``.

    Suppose that ``b`` is equivalent to ``cupy.tensordot(a, x)``.
    This function computes tensor ``x`` from ``a`` and ``b``.

    Args:
        a (cupy.ndarray): The tensor with ``len(shape) >= 1``
        b (cupy.ndarray): The tensor with ``len(shape) >= 1``
        axes (tuple of ints): Axes in ``a`` to reorder to the right
            before inversion.

    Returns:
        cupy.ndarray:
            The tensor with shape ``Q`` such that ``b.shape + Q == a.shape``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.tensorsolve`
    """
    if axes is not None:
        allaxes = list(range(a.ndim))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(a.ndim, k)
        a = a.transpose(allaxes)

    oldshape = a.shape[-(a.ndim - b.ndim):]
    prod = internal.prod(oldshape)

    a = a.reshape(-1, prod)
    b = b.ravel()
    result = solve(a, b)
    return result.reshape(oldshape)


def _nrm2_last_axis(x):
    real_dtype = x.dtype.char.lower()
    x = cupy.ascontiguousarray(x)
    return cupy.sum(cupy.square(x.view(real_dtype)), axis=-1)


def lstsq(a, b, rcond='warn'):
    """Return the least-squares solution to a linear matrix equation.

    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.

    Args:
        a (cupy.ndarray): "Coefficient" matrix with dimension ``(M, N)``
        b (cupy.ndarray): "Dependent variable" values with dimension ``(M,)``
            or ``(M, K)``
        rcond (float): Cutoff parameter for small singular values.
            For stability it computes the largest singular value denoted by
            ``s``, and sets all singular values smaller than ``s`` to zero.

    Returns:
        tuple:
            A tuple of ``(x, residuals, rank, s)``. Note ``x`` is the
            least-squares solution with shape ``(N,)`` or ``(N, K)`` depending
            if ``b`` was two-dimensional. The sums of ``residuals`` is the
            squared Euclidean 2-norm for each column in b - a*x. The
            ``residuals`` is an empty array if the rank of a is < N or M <= N,
            but  iff b is 1-dimensional, this is a (1,) shape array, Otherwise
            the shape is (K,). The ``rank`` of matrix ``a`` is an integer. The
            singular values of ``a`` are ``s``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.lstsq`
    """
    if rcond == 'warn':
        warnings.warn(
            '`rcond` parameter will change to the default of '
            'machine precision times ``max(M, N)`` where M and N '
            'are the input matrix dimensions.\n'
            'To use the future default and silence this warning '
            'we advise to pass `rcond=None`, to keep using the old, '
            'explicitly pass `rcond=-1`.',
            FutureWarning)
        rcond = -1

    _util._assert_cupy_array(a, b)
    _util._assert_rank2(a)
    if b.ndim > 2:
        raise linalg.LinAlgError('{}-dimensional array given. Array must be at'
                                 ' most two-dimensional'.format(b.ndim))
    m, n = a.shape[-2:]
    m2 = b.shape[0]
    if m != m2:
        raise linalg.LinAlgError('Incompatible dimensions')

    u, s, vh = cupy.linalg.svd(a, full_matrices=False)

    if rcond is None:
        rcond = numpy.finfo(s.dtype).eps * max(m, n)
    elif rcond <= 0 or rcond >= 1:
        # some doc of gelss/gelsd says "rcond < 0", but it's not true!
        rcond = numpy.finfo(s.dtype).eps

    # number of singular values and matrix rank
    cutoff = rcond * s.max()
    s1 = 1 / s
    sing_vals = s <= cutoff
    s1[sing_vals] = 0
    rank = s.size - sing_vals.sum(dtype=numpy.int32)

    # Solve the least-squares solution
    # x = vh.T.conj() @ diag(s1) @ u.T.conj() @ b
    z = (cupy.dot(b.T, u.conj()) * s1).T
    x = cupy.dot(vh.T.conj(), z)
    # Calculate squared Euclidean 2-norm for each column in b - a*x
    if m <= n or rank != n:
        resids = cupy.empty((0,), dtype=s.dtype)
    else:
        e = b - a.dot(x)
        resids = cupy.atleast_1d(_nrm2_last_axis(e.T))
    return x, resids, rank, s


def inv(a):
    """Computes the inverse of a matrix.

    This function computes matrix ``a_inv`` from n-dimensional regular matrix
    ``a`` such that ``dot(a, a_inv) == eye(n)``.

    Args:
        a (cupy.ndarray): The regular matrix

    Returns:
        cupy.ndarray: The inverse of a matrix.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.inv`
    """
    if a.ndim >= 3:
        return _batched_inv(a)

    _util._assert_cupy_array(a)
    _util._assert_rank2(a)
    _util._assert_nd_squareness(a)

    dtype, out_dtype = _util.linalg_common_type(a)
    order = 'F' if a._f_contiguous else 'C'
    # prevent 'a' to be overwritten
    a = a.astype(dtype, copy=True, order=order)
    b = cupy.eye(a.shape[0], dtype=dtype, order=order)
    if order == 'F':
        cupyx.lapack.gesv(a, b)
    else:
        cupyx.lapack.gesv(a.T, b.T)
    return b.astype(out_dtype, copy=False)


def _batched_inv(a):

    assert(a.ndim >= 3)
    _util._assert_cupy_array(a)
    _util._assert_nd_squareness(a)
    dtype, out_dtype = _util.linalg_common_type(a)

    if dtype == cupy.float32:
        getrf = cupy.cuda.cublas.sgetrfBatched
        getri = cupy.cuda.cublas.sgetriBatched
    elif dtype == cupy.float64:
        getrf = cupy.cuda.cublas.dgetrfBatched
        getri = cupy.cuda.cublas.dgetriBatched
    elif dtype == cupy.complex64:
        getrf = cupy.cuda.cublas.cgetrfBatched
        getri = cupy.cuda.cublas.cgetriBatched
    elif dtype == cupy.complex128:
        getrf = cupy.cuda.cublas.zgetrfBatched
        getri = cupy.cuda.cublas.zgetriBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    if 0 in a.shape:
        return cupy.empty_like(a, dtype=out_dtype)
    a_shape = a.shape

    # copy is necessary to present `a` to be overwritten.
    a = a.astype(dtype, order='C').reshape(-1, a_shape[-2], a_shape[-1])

    handle = device.get_cublas_handle()
    batch_size = a.shape[0]
    n = a.shape[1]
    lda = n
    step = n * lda * a.itemsize
    start = a.data.ptr
    stop = start + step * batch_size
    a_array = cupy.arange(start, stop, step, dtype=cupy.uintp)
    pivot_array = cupy.empty((batch_size, n), dtype=cupy.int32)
    info_array = cupy.empty((batch_size,), dtype=cupy.int32)

    getrf(handle, n, a_array.data.ptr, lda, pivot_array.data.ptr,
          info_array.data.ptr, batch_size)
    cupy.linalg._util._check_cublas_info_array_if_synchronization_allowed(
        getrf, info_array)

    c = cupy.empty_like(a)
    ldc = lda
    step = n * ldc * c.itemsize
    start = c.data.ptr
    stop = start + step * batch_size
    c_array = cupy.arange(start, stop, step, dtype=cupy.uintp)

    getri(handle, n, a_array.data.ptr, lda, pivot_array.data.ptr,
          c_array.data.ptr, ldc, info_array.data.ptr, batch_size)
    cupy.linalg._util._check_cublas_info_array_if_synchronization_allowed(
        getri, info_array)

    return c.reshape(a_shape).astype(out_dtype, copy=False)


# TODO(leofang): support the hermitian keyword?
def pinv(a, rcond=1e-15):
    """Compute the Moore-Penrose pseudoinverse of a matrix.

    It computes a pseudoinverse of a matrix ``a``, which is a generalization
    of the inverse matrix with Singular Value Decomposition (SVD).
    Note that it automatically removes small singular values for stability.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(..., M, N)``
        rcond (float or cupy.ndarray): Cutoff parameter for small singular
            values. For stability it computes the largest singular value
            denoted by ``s``, and sets all singular values smaller than
            ``rcond * s`` to zero. Broadcasts against the stack of matrices.

    Returns:
        cupy.ndarray: The pseudoinverse of ``a`` with dimension
            ``(..., N, M)``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.pinv`
    """
    _util._assert_cupy_array(a)
    if a.size == 0:
        _, out_dtype = _util.linalg_common_type(a)
        m, n = a.shape[-2:]
        if m == 0 or n == 0:
            out_dtype = a.dtype  # NumPy bug?
        return cupy.empty(a.shape[:-2] + (n, m), dtype=out_dtype)

    u, s, vt = _decomposition.svd(a.conj(), full_matrices=False)

    # discard small singular values
    cutoff = rcond * cupy.amax(s, axis=-1)
    leq = s <= cutoff[..., None]
    cupy.reciprocal(s, out=s)
    s[leq] = 0

    return cupy.matmul(vt.swapaxes(-2, -1), s[..., None] * u.swapaxes(-2, -1))


def tensorinv(a, ind=2):
    """Computes the inverse of a tensor.

    This function computes tensor ``a_inv`` from tensor ``a`` such that
    ``tensordot(a_inv, a, ind) == I``, where ``I`` denotes the identity tensor.

    Args:
        a (cupy.ndarray):
            The tensor such that
            ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
        ind (int):
            The positive number used in ``axes`` option of ``tensordot``.

    Returns:
        cupy.ndarray:
            The inverse of a tensor whose shape is equivalent to
            ``a.shape[ind:] + a.shape[:ind]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. seealso:: :func:`numpy.linalg.tensorinv`
    """
    _util._assert_cupy_array(a)

    if ind <= 0:
        raise ValueError('Invalid ind argument')
    oldshape = a.shape
    invshape = oldshape[ind:] + oldshape[:ind]
    prod = internal.prod(oldshape[ind:])
    a = a.reshape(prod, -1)
    a_inv = inv(a)
    return a_inv.reshape(*invshape)
