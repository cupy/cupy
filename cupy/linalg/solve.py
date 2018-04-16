import numpy
from numpy import linalg
import six

import cupy
from cupy.core import core
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import decomposition
from cupy.linalg import util

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


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

    .. seealso:: :func:`numpy.linalg.solve`
    """
    # NOTE: Since cusolver in CUDA 8.0 does not support gesv,
    #       we manually solve a linear system with QR decomposition.
    #       For details, please see the following:
    #       https://docs.nvidia.com/cuda/cusolver/index.html#qr_examples
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    util._assert_cupy_array(a, b)
    util._assert_nd_squareness(a)

    if not ((a.ndim == b.ndim or a.ndim == b.ndim + 1) and
            a.shape[:-1] == b.shape[:a.ndim - 1]):
        raise ValueError(
            'a must have (..., M, M) shape and b must have (..., M) '
            'or (..., M, K)')

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ())

    a = a.astype(dtype)
    b = b.astype(dtype)
    if a.ndim == 2:
        return _solve(a, b)
    x = cupy.empty_like(b)
    shape = a.shape[:-2]
    for i in six.moves.range(numpy.prod(shape)):
        index = numpy.unravel_index(i, shape)
        x[index] = _solve(a[index], b[index])
    return x


def _solve(a, b):
    a = cupy.asfortranarray(a)
    b = cupy.asfortranarray(b)
    dtype = a.dtype
    m, k = (b.size, 1) if b.ndim == 1 else b.shape
    cusolver_handle = device.get_cusolver_handle()
    cublas_handle = device.get_cublas_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        geqrf = cusolver.sgeqrf
        geqrf_bufferSize = cusolver.sgeqrf_bufferSize
        ormqr = cusolver.sormqr
        trsm = cublas.strsm
    else:  # dtype == 'd'
        geqrf = cusolver.dgeqrf
        geqrf_bufferSize = cusolver.dgeqrf_bufferSize
        ormqr = cusolver.dormqr
        trsm = cublas.dtrsm

    # 1. QR decomposition (A = Q * R)
    buffersize = geqrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)
    tau = cupy.empty(m, dtype=dtype)
    geqrf(
        cusolver_handle, m, m, a.data.ptr, m,
        tau.data.ptr, workspace.data.ptr, buffersize, dev_info.data.ptr)
    _check_status(dev_info)
    # 2. ormqr (Q^T * B)
    ormqr(
        cusolver_handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_OP_T,
        m, k, m, a.data.ptr, m, tau.data.ptr, b.data.ptr, m,
        workspace.data.ptr, buffersize, dev_info.data.ptr)
    _check_status(dev_info)
    # 3. trsm (X = R^{-1} * (Q^T * B))
    trsm(
        cublas_handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
        cublas.CUBLAS_OP_N, cublas.CUBLAS_DIAG_NON_UNIT,
        m, k, 1, a.data.ptr, m, b.data.ptr, m)
    return b


def _check_status(dev_info):
    status = int(dev_info)
    if status < 0:
        raise linalg.LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')


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

    .. seealso:: :func:`numpy.linalg.tensorsolve`
    """
    if axes is not None:
        allaxes = list(six.moves.range(a.ndim))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(a.ndim, k)
        a = a.transpose(allaxes)

    oldshape = a.shape[-(a.ndim - b.ndim):]
    prod = cupy.internal.prod(oldshape)

    a = a.reshape(-1, prod)
    b = b.ravel()
    result = solve(a, b)
    return result.reshape(oldshape)


# TODO(okuta): Implement lstsq


def inv(a):
    """Computes the inverse of a matrix.

    This function computes matrix ``a_inv`` from n-dimensional regular matrix
    ``a`` such that ``dot(a, a_inv) == eye(n)``.

    Args:
        a (cupy.ndarray): The regular matrix

    Returns:
        cupy.ndarray: The inverse of a matrix.

    .. seealso:: :func:`numpy.linalg.inv`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # to prevent `a` to be overwritten
    a = a.copy()

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=dtype)

    ipiv = cupy.empty((a.shape[0], 1), dtype=dtype)

    if dtype == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
        getrs = cusolver.sgetrs
    else:  # dtype == 'd'
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize
        getrs = cusolver.dgetrs

    m = a.shape[0]

    buffersize = getrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)

    # LU factorization
    getrf(cusolver_handle, m, m, a.data.ptr, m, workspace.data.ptr,
          ipiv.data.ptr, dev_info.data.ptr)

    b = cupy.eye(m, dtype=dtype)

    # solve for the inverse
    getrs(cusolver_handle, 0, m, m, a.data.ptr, m, ipiv.data.ptr, b.data.ptr,
          m, dev_info.data.ptr)

    return b


def pinv(a, rcond=1e-15):
    """Compute the Moore-Penrose pseudoinverse of a matrix.

    It computes a pseudoinverse of a matrix ``a``, which is a generalization
    of the inverse matrix with Singular Value Decomposition (SVD).
    Note that it automatically removes small singular values for stability.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, N)``
        rcond (float): Cutoff parameter for small singular values.
            For stability it computes the largest singular value denoted by
            ``s``, and sets all singular values smaller than ``s`` to zero.

    Returns:
        cupy.ndarray: The pseudoinverse of ``a`` with dimension ``(N, M)``.

    .. seealso:: :func:`numpy.linalg.pinv`
    """
    u, s, vt = decomposition.svd(a, full_matrices=False)
    cutoff = rcond * s.max()
    s1 = 1 / s
    s1[s <= cutoff] = 0
    return core.dot(vt.T, s1[:, None] * u.T)


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

    .. seealso:: :func:`numpy.linalg.tensorinv`
    """
    util._assert_cupy_array(a)

    if ind <= 0:
        raise ValueError('Invalid ind argument')
    oldshape = a.shape
    invshape = oldshape[ind:] + oldshape[:ind]
    prod = cupy.internal.prod(oldshape[ind:])
    a = a.reshape(prod, -1)
    a_inv = inv(a)
    return a_inv.reshape(*invshape)
