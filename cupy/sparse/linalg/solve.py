import numpy

import cupy
from cupy import cuda
from cupy.cuda import device
from cupy.linalg import util
from cupy.sparse import csr_matrix
from cupy.sparse import isspmatrix_csr


if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def lsqr(A, b):

    """Solve linear system with QR decomposition.

    Find the solution to a large, sparse, linear system of equations.
    Given two-dimensional matrix `A` is decomposed into ``Q * R``.

    Args:
        A (cupy.ndarray or cupy.sparse.csr_matrix): The input matrix with
            dimension ``(N, N)``
        b (cupy.ndarray): Right-hand side vector.

    Returns:
        x (cupy.ndarray): The final solution.

    .. seealso:: :func:`scipy.sparse.linalg.lsqr`
    """

    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    if not isspmatrix_csr(A):
        A = csr_matrix(A)
    util._assert_nd_squareness(A)
    util._assert_cupy_array(b)
    m, _ = A.shape
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')

    # Cast to float32 or float64
    if A.dtype.char == 'f' or A.dtype.char == 'd':
        dtype = A.dtype.char
    else:
        dtype = numpy.find_common_type((A.dtype.char, 'f'), ()).char

    handle = device.get_cusolver_handlesp()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cupy.empty(m, dtype=dtype)

    if dtype == 'f':
        singularity = numpy.empty(1, numpy.int32)
        cusolver.scsrlsvqr(
            handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
            A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
            x.data.ptr, singularity.ctypes.data)
    else:  # dtype == 'd'
        singularity = numpy.empty(1, numpy.int64)
        cusolver.dcsrlsvqr(
            handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
            A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
            x.data.ptr, singularity.ctypes.data)

    return x
