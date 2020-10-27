import numpy

import cupy
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.linalg import _util
from cupyx.scipy import sparse


def lschol(A, b):
    """Solves linear system with cholesky decomposition.

    Find the solution to a large, sparse, linear system of equations.
    The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is
    decomposed into ``L * L^*``.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix
            with dimension ``(N, N)``. Must be positive-definite input matrix.
            Only symmetric real matrix is supported currently.
        b (cupy.ndarray): Right-hand side vector.

    Returns:
        ret (cupy.ndarray): The solution vector ``x``.

    """

    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    _util._assert_nd_squareness(A)
    _util._assert_cupy_array(b)
    m = A.shape[0]
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')

    # Cast to float32 or float64
    if A.dtype == 'f' or A.dtype == 'd':
        dtype = A.dtype
    else:
        dtype = numpy.promote_types(A.dtype, 'f')

    handle = device.get_cusolver_sp_handle()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cupy.empty(m, dtype=dtype)
    singularity = numpy.empty(1, numpy.int32)

    if dtype == 'f':
        csrlsvchol = cusolver.scsrlsvchol
    else:
        csrlsvchol = cusolver.dcsrlsvchol
    csrlsvchol(
        handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
        A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
        x.data.ptr, singularity.ctypes.data)

    # The return type of SciPy is always float64.
    x = x.astype(numpy.float64)

    return x
