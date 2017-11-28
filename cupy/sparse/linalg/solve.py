import numpy

import cupy
from cupy import cuda
from cupy.cuda import device
from cupy.linalg import util
from cupy import sparse


if cuda.cusolver_enabled:
    from cupy.cuda import cusolver


def lsqr(A, b):
    """Solves linear system with QR decomposition.

    Find the solution to a large, sparse, linear system of equations.
    The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is
    decomposed into ``Q * R``.

    Args:
        A (cupy.ndarray or cupy.sparse.csr_matrix): The input matrix with
            dimension ``(N, N)``
        b (cupy.ndarray): Right-hand side vector.

    Returns:
        ret (tuple): Its length must be ten. It has same type elements
            as SciPy. Only first, fourth and ninth elements are available.
            The first element is the solution vector, ``x``.
            The fourth element is ``norm(b - Ax)``.
            The ninth element is ``norm(x)``.
            Other elements are expressed as ``None`` because the
            implementation of cuSOLVER is different from the one of SciPy.

    .. seealso:: :func:`scipy.sparse.linalg.lsqr`
    """

    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    util._assert_nd_squareness(A)
    util._assert_cupy_array(b)
    m = A.shape[0]
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')

    # Cast to float32 or float64
    if A.dtype == 'f' or A.dtype == 'd':
        dtype = A.dtype
    else:
        dtype = numpy.find_common_type((A.dtype, 'f'), ())

    handle = device.get_cusolver_sp_handle()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cupy.empty(m, dtype=dtype)
    singularity = numpy.empty(1, numpy.int32)

    if dtype == 'f':
        csrlsvqr = cusolver.scsrlsvqr
    else:
        csrlsvqr = cusolver.dcsrlsvqr
    csrlsvqr(
        handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
        A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
        x.data.ptr, singularity.ctypes.data)

    r1norm = cupy.linalg.norm(b - A.dot(x))
    xnorm = cupy.linalg.norm(x)
    # The return type of SciPy is always float64. Therefore, x must be casted.
    x = x.astype(numpy.float64)
    ret = (x, None, None, r1norm, None, None, None, None, xnorm, None)
    return ret
