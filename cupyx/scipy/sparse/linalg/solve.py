import numpy

import cupy
from cupy.cuda import cusolver, device
from cupy.linalg import util
import cupy.cusparse
import cupyx.scipy.sparse
from cupy_backends.cuda.libs import cusparse


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
        dtype = numpy.promote_types(A.dtype, 'f')

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

    # The return type of SciPy is always float64. Therefore, x must be casted.
    x = x.astype(numpy.float64)
    ret = (x, None, None, None, None, None, None, None, None, None)
    return ret


def bicgstab(A, b, x0=None, M=None, tol=1e-5, maxiter=numpy.inf,
             callback=None, atol='legacy'):
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
        M (callable, str): GPU preconditioner function or str to indicate
                            preconditioner choice, can also specify
                            preconditioner as a string (e.g., 'ilu', 'jacobi').
        tol (float): Tolerance for convergence, exits when
                     residual norm(r) < tol * norm(b)
        maxiter (int): maximum number of iterations (defaults to `numpy.inf`)
        callback (callable): a callback function to call at each iteration
                            (unlike scipy, it accepts iteration and
                            residual as well as current vector `x`)
        atol (str): Currently a dummy parameter to be used in a future release

    Returns:
        tuple:
            A tuple of result x and info (as in scipy, 0 if successful,
            1 if no convergence, <0 if rho or omega breakdown)

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
        dtype = numpy.promote_types(A.dtype.char, 'f').char

    # Calculate initial residual
    x0 = x0 if x0 is not None else cupy.zeros(n, dtype=dtype)
    r = cupy.cusparse.spmv(A, x0, b, -1, 1)
    p, q = cupy.zeros_like(r), cupy.zeros_like(r)
    r_tilde = r.copy()

    # Main BiCGSTAB loop

    i = 0
    rho = alpha = omega = info = 1

    x = x0.copy()
    tol = cupy.linalg.norm(b) * tol
    atol = 0 if atol == 'legacy' else atol

    # preconditioner
    M, M_info = get_M(A, M)

    while i < maxiter:
        i += 1
        prev_rho = rho
        rho = cupy.dot(r_tilde.conj(), r)
        if rho == 0:
            info = -10
            break
        beta = rho / prev_rho * alpha / omega
        p = r + beta * (p - omega * q)

        # Alpha step
        phat = M(p)  # apply preconditioner

        q = cupy.cusparse.spmv(A, phat)
        alpha = rho / cupy.dot(r_tilde.conj(), q)
        s = r - alpha * q

        residual = cupy.linalg.norm(s)
        if residual < tol or residual < atol:
            x += alpha * phat
            info = 0
            break

        # Omega step
        shat = M(s)  # apply preconditioner

        t = cupy.cusparse.spmv(A, shat)
        omega = cupy.dot(cupy.conj(t), s) / cupy.square(cupy.linalg.norm(t))
        if omega == 0:
            info = -11
            break

        x += alpha * phat + omega * shat
        r = s - omega * t

        residual = cupy.linalg.norm(r)
        if callback is not None:
            callback(x, i, r)
        if residual < tol or residual < atol:
            info = 0
            break

    _destroy_ilu(M_info)

    return x, info


def get_M(A, preconditioner=None):
    """

    Args:
        A: Matrix to be preconditioned
        preconditioner: callable, nparray, or str to specify preconditioner method

    Returns:

    """
    if hasattr(preconditioner, '__call__'):
        return preconditioner, None
    elif isinstance(preconditioner, cupy.ndarray):
        ndim = preconditioner.ndim
        n = preconditioner.shape[0]
        if not ndim == 1 and n == A.n:
            expect_msg = 'Expected M.ndim == 1 and M.shape[0] == {}, '.format(A.n)
            actual_msg = 'got M.ndim == {} and M.shape[0] == {}'.format(n, ndim)
            raise ValueError(expect_msg, actual_msg)
        M = (lambda vec: vec / M), None
    elif preconditioner == 'ilu':
        return _get_M_ilu(A.copy())
    elif preconditioner == 'jacobi':
        raise NotImplementedError(
            'A.diagonal() still needs to be implemented...')
        # M = lambda vec: vec / A.diagonal(), None, None, None
    elif preconditioner is None:
        return (lambda vec: vec), None
    else:
        raise AttributeError('Expected M to be ilu, jacobi or cupy.ndarray')


def _get_M_ilu(A):
    # Setup M, the Incomplete LU factorization of A
    # See https://docs.nvidia.com/cuda/cusparse/#csrilu02_solve

    # support float32, float64, complex64, and complex128
    handle = device.get_cusolver_sp_handle()

    if A.dtype.char in 'fdFD':
        dtype = A.dtype.char
    else:
        dtype = numpy.promote_types(A.dtype.char, 'f').char
    alpha = numpy.array(1, dtype).ctypes

    def A_tuple(descr):
        return (A.shape[0], A.nnz, descr.descriptor, A.data.data.ptr,
                A.indptr.data.ptr, A.indices.data.ptr)

    def A_tuple_a(descr):
        return (A.shape[0], A.nnz, alpha.data, descr.descriptor,
                A.data.data.ptr, A.indptr.data.ptr, A.indices.data.ptr)

    n = A.shape[0]

    # create info objects
    info_M = cusparse.createCsrilu02Info()
    info_L, info_U = cusparse.createCsrsv2Info(
    ), cusparse.createCsrsv2Info()

    # create solve policies
    policy_M = cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    policy_L = cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    policy_U = cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL

    # define trans
    trans_L = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    trans_U = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

    # create descriptions
    descr_M = cupy.cusparse.MatDescriptor.create()
    descr_L = cupy.cusparse.MatDescriptor.create()
    descr_U = cupy.cusparse.MatDescriptor.create()
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
    buff_size_M = cupy.cusparse._call_cusparse(
        'csrilu02_bufferSize', dtype, handle, *A_tuple(descr_M), info_M)
    buff_size_L = cupy.cusparse._call_cusparse(
        'csrsv2_bufferSize', dtype, handle, trans_L, *A_tuple(descr_L), info_L)
    buff_size_U = cupy.cusparse._call_cusparse(
        'csrsv2_bufferSize', dtype, handle, trans_U, *A_tuple(descr_U), info_U)
    buff_size = numpy.max((buff_size_M, buff_size_L, buff_size_U))
    buff = cupy.empty(int(buff_size), dtype=cupy.int8)
    assert buff.data.ptr % 128 == 0

    # Analysis (setup) of M = LU, the Incomplete LU factorization of A
    cupy.cusparse._call_cusparse('csrilu02_analysis', dtype, handle,
                            *A_tuple(descr_M), info_M, policy_M, buff.data.ptr)
    cupy.cusparse._call_cusparse('csrsv2_analysis', dtype, handle, trans_L,
                            *A_tuple(descr_L), info_L, policy_L, buff.data.ptr)
    cupy.cusparse._call_cusparse('csrsv2_analysis', dtype, handle, trans_U,
                            *A_tuple(descr_U), info_U, policy_U, buff.data.ptr)

    # Perform M = L * U incomplete decomposition
    cupy.cusparse._call_cusparse('csrilu02', dtype, handle,
                            *A_tuple(descr_M), info_M, policy_M, buff.data.ptr)

    y = cupy.empty(n, dtype=dtype)

    def M(x):
        out = cupy.empty(n, dtype=dtype)
        cupy.cusparse._call_cusparse('csrsv2_solve', dtype, handle, trans_L,
                                *A_tuple_a(descr_L), info_L, x.data.ptr,
                                y.data.ptr, info_L, policy_L, buff.data.ptr)
        cupy.cusparse._call_cusparse('csrsv2_solve', dtype, handle, trans_U,
                                *A_tuple_a(descr_U), info_U, y.data.ptr,
                                out.data.ptr, info_U, policy_U, buff.data.ptr)
        return out

    return M, (info_M, info_U, info_L)


def _destroy_ilu(info_list):
    if info_list is not None:
        for info in info_list:
            cusparse.destroyCsrilu02Info(info)
