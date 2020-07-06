import numpy as np

import cupy as cp
from cupy.cuda import cusolver, device, cublas, cusparse
from cupy.linalg import util
from cupy import cusparse
import cupyx.scipy.sparse


def lsqr(A, b):
    """Solves linear system with QR decomposition.

    Find the solution to a large, sparse, linear system of equations.
    The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is
    decomposed into ``Q * R``.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix
            with dimension ``(N, N)``
        b (cupy.ndarray): Right-hand side vector.

    Returns:
        tuple:
            Its length must be ten. It has same type elements
            as SciPy. Only the first element, the solution vector ``x``, is
            available and other elements are expressed as ``None`` because
            the implementation of cuSOLVER is different from the one of SciPy.
            You can easily calculate the fourth element by ``norm(b - Ax)``
            and the ninth element by ``norm(x)``.

    .. seealso:: :func:`scipy.sparse.linalg.lsqr`
    """

    if not cupyx.scipy.sparse.isspmatrix_csr(A):
        A = cupyx.scipy.sparse.csr_matrix(A)
    util._assert_nd_squareness(A)
    util._assert_cupy_array(b)
    m = A.shape[0]
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')

    # Cast to float32 or float64
    if A.dtype == 'f' or A.dtype == 'd':
        dtype = A.dtype
    else:
        dtype = np.promote_types(A.dtype, 'f')

    handle = device.get_cusolver_sp_handle()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cp.empty(m, dtype=dtype)
    singularity = np.empty(1, np.int32)

    if dtype == 'f':
        csrlsvqr = cusolver.scsrlsvqr
    else:
        csrlsvqr = cusolver.dcsrlsvqr
    csrlsvqr(
        handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
        A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
        x.data.ptr, singularity.ctypes.data)

    # The return type of SciPy is always float64. Therefore, x must be casted.
    x = x.astype(np.float64)
    ret = (x, None, None, None, None, None, None, None, None, None)
    return ret


def bicgstab(A, b, x0=None, M='ilu', tol=1e-5, maxit=np.inf):
    """Solves linear system using an iterative solver BiCGSTAB.

    See https://docs.nvidia.com/cuda/cusparse/#csrilu02_solve and
    https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html.

    Find the solution to a large, sparse, linear system of equations.
    The function approximates ``Ax = b`` using BiCGSTAB iterative method.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix
            with dimension ``(N, N)``
        b (cupy.ndarray): Right-hand side vector.
        x0 (cupy.ndarray): Initial guess (defaults to zeros)
        M (cupy.ndarray, str, callable): preconditioner (diagonal if ndarray, or specified by string)
        maxit (int): maximum number of iterations (defaults to `np.inf`)

    Returns:
        tuple:
            A tuple of result x and info (always 0)

    .. seealso:: :func:`scipy.sparse.linalg.bicgstab`
    """

    if not cupyx.scipy.sparse.isspmatrix_csr(A):
        A = cupyx.scipy.sparse.csr_matrix(A)
    util._assert_nd_squareness(A)
    util._assert_cupy_array(b)
    n = A.shape[0]
    if b.ndim != 1 or len(b) != n:
        raise ValueError('b must be 1-d array whose size is same as A')

    # support float32, float64, complex64, and complex128
    if A.dtype.char in 'fdFD':
        dtype = A.dtype.char
    else:
        dtype = np.promote_types(A.dtype.char, 'f').char

    handle = device.get_cusolver_sp_handle()

    if M == 'ilu':
        msolve = _get_msolve_ilu(A, handle)
    elif M == 'jacobi':
        msolve = lambda vec: vec / A.diagonal()
    elif isinstance(M, cp.ndarray):
        if not M.ndim == 1 and M.shape[0] == n:
            raise ValueError(f'Expected M.ndim == 1 and M.shape[0] == {n},'
                             f'but got M.ndim == {M.ndim} and M.shape[0] == {M.shape[0]}')
        msolve = lambda vec: vec / M
    else:
        raise AttributeError('Expected M to be ilu, jacobi or cp.ndarray')

    # Calculate initial residual
    x0 = x0.copy() if x0 is not None else cp.zeros(n, dtype=dtype)
    r = cusparse.spmv(A, -x0, b, 1, 1)
    p, q = cp.zeros_like(r), cp.zeros_like(r)
    r_tilde = r.copy()

    # Main BiCGSTAB loop

    i = 0
    rho = 1
    phat, shat = cp.empty_like(r), cp.empty_like(r)
    prev_rho, alpha, omega = None, None, None

    x = x0

    while i < maxit:
        i += 1
        prev_rho = rho
        rho = cp.dot(r_tilde, r)
        if rho == 0:
            raise RuntimeError('rho = 0')
        beta = rho / prev_rho * alpha / omega
        p = r + beta * (p - omega * q)

        # Alpha step
        if M == 'ilu':
            msolve(p, phat)
        else:
            phat = msolve(p)

        q = cusparse.spmv(A, phat)
        alpha = rho / cp.dot(r_tilde, q)
        s = r - alpha * q
        x = x + alpha * phat

        if cp.dot(s, s) < tol ** 2:
            break

        # Omega step
        if M == 'ilu':
            msolve(s, shat)
        else:
            shat = msolve(s)
        t = cusparse.spmv(A, shat)
        omega = cp.dot(t, s) / cp.dot(t, t)
        x = x + omega * s
        r = s - omega * t

    return x, 0


def _get_msolve_ilu(A, handle):
    # Setup M, the Incomplete LU factorization of A
    # See https://docs.nvidia.com/cuda/cusparse/#csrilu02_solve

    # support float32, float64, complex64, and complex128
    if A.dtype.char in 'fdFD':
        dtype = A.dtype.char
    else:
        dtype = np.promote_types(A.dtype.char, 'f').char
    alpha = np.array(1, dtype).ctypes
    A_tuple = lambda descr: (A.shape[0], A.nnz, descr.descriptor, A.data.data.ptr,
                             A.indptr.data.ptr, A.indices.data.ptr)
    A_tuple_a = lambda descr: (A.shape[0], A.nnz, alpha.data, descr.descriptor, A.data.data.ptr,
                               A.indptr.data.ptr, A.indices.data.ptr)
    n = A.shape[0]

    # create info objects
    info_M = cusparse.createCsrilu02Info()
    info_L, info_U = cusparse.createCsrsv2Info(), cusparse.createCsrsv2Info()

    # create solve policies
    policy_M, policy_L = cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL, cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    policy_U = cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL

    # define trans
    trans_L, trans_U = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

    # create descriptions
    descr_M, descr_L, descr_U = cusparse.MatDescriptor.create(), cusparse.MatDescriptor.create(), \
                                cusparse.MatDescriptor.create()
    descr_M.set_mat_index_base(cusparse.CUSPARSE_INDEX_BASE_ONE)
    descr_M.set_mat_type(cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    descr_L.set_mat_index_base(cusparse.CUSPARSE_INDEX_BASE_ONE)
    descr_L.set_mat_type(cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    descr_L.set_mat_fill_mode(cusparse.CUSPARSE_FILL_MODE_LOWER)
    descr_L.set_mat_diag_type(cusparse.CUSPARSE_DIAG_TYPE_UNIT)
    descr_U.set_mat_index_base(cusparse.CUSPARSE_INDEX_BASE_ONE)
    descr_U.set_mat_type(cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    descr_U.set_mat_fill_mode(cusparse.CUSPARSE_FILL_MODE_UPPER)
    descr_U.set_mat_diag_type(cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT)

    # get required buffer size and allocate corresponding memory
    buff_size_M = cusparse._call_cusparse('csrilu02_bufferSize', dtype, handle,
                                          *A_tuple(descr_M), info_M.ptr)
    buff_size_L = cusparse._call_cusparse('csrsv2_bufferSize', dtype, handle,
                                          *A_tuple(descr_L), info_L.ptr)
    buff_size_U = cusparse._call_cusparse('csrsv2_bufferSize', dtype, handle,
                                          *A_tuple(descr_U), info_U.ptr)
    buff_size = np.max((buff_size_M, buff_size_L, buff_size_U))
    buff = cp.empty(buff_size, np.int8)

    # Analysis (setup) of M = LU, the Incomplete LU factorization of A
    cusparse._call_cusparse('csrilu02_analysis', dtype, handle,
                            *A_tuple(descr_M), info_M.ptr, policy_M, buff.data.ptr)
    cusparse._call_cusparse('csrsv2_analysis', dtype, handle, trans_L,
                            *A_tuple(descr_L), info_L.ptr, policy_L, buff.data.ptr)
    cusparse._call_cusparse('csrsv2_analysis', dtype, handle, trans_U,
                            *A_tuple(descr_U), info_U.ptr, policy_U, buff.data.ptr)

    y = cp.empty(n, dtype=dtype)

    def msolve(vec, out):
        cusparse._call_cusparse('csrsv2_solve', dtype, handle, trans_L, *A_tuple_a(descr_L),
                                info_L.ptr, vec.data.ptr, y.data.ptr, policy_L, buff.data.ptr)
        cusparse._call_cusparse('csrsv2_solve', dtype, handle, trans_U, *A_tuple_a(descr_U),
                                info_L.ptr, y.data.ptr, out.data.ptr, policy_U, buff.data.ptr)

    return msolve
