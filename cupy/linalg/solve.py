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
        a (cupy.ndarray): The matrix with dimension ``(M, M)``
        b (cupy.ndarray): The vector with ``M`` elements, or
            the matrix with dimension ``(M, K)``

    Returns:
        cupy.ndarray:
            The vector with ``M`` elements, or the matrix with dimension
            ``(M, K)``.

    .. seealso:: :func:`numpy.linalg.solve`
    """
    # NOTE: Since cusolver in CUDA 8.0 does not support gesv,
    #       we manually solve a linear system with QR decomposition.
    #       For details, please see the following:
    #       http://docs.nvidia.com/cuda/cusolver/index.html#qr_examples
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    util._assert_cupy_array(a, b)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)
    if 2 < b.ndim:
        raise linalg.LinAlgError(
            '{}-dimensional array given. Array must be '
            'one or two-dimensional'.format(b.ndim))
    if len(a) != len(b):
        raise linalg.LinAlgError(
            'The number of rows of array a must be '
            'the same as that of array b')

    # Cast to float32 or float64
    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.find_common_type((a.dtype.char, 'f'), ()).char

    m, k = (b.size, 1) if b.ndim == 1 else b.shape
    a = a.transpose().astype(dtype, order='C', copy=True)
    b = b.transpose().astype(dtype, order='C', copy=True)
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
    return b.transpose()


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
    prod = numpy.prod(oldshape)

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

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    b = cupy.eye(len(a), dtype=a.dtype)
    return solve(a, b)


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


# TODO(okuta): Implement tensorinv
