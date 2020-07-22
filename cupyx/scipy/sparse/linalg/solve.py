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
        M (tuple(callable, optional(info))): GPU preconditioner function
            and info object for preconditioner (to destroy after running)
        tol (float): Tolerance for convergence, exits when
                     residual norm(r) < tol * norm(b)
        maxiter (int): maximum number of iterations (defaults to `numpy.inf`)
        callback (callable): a callback function to call at each iteration
                            (unlike scipy, it accepts iteration and
                            residual as well as current vector `x`)
        atol (float): Default to 0 (use tol instead of atol for convergence)

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
    rho = alpha = omega = cupy.linalg.norm(r)

    info = 1

    x = x0.copy()
    tol = cupy.linalg.norm(b) * tol
    atol = 0.0 if atol == 'legacy' else atol

    # preconditioner
    M, M_info = (_identity, None) if M is None else M

    while i < maxiter:
        i += 1
        prev_rho = rho
        rho = cupy.vdot(r_tilde, r)
        if rho == 0:
            info = -10
            break
        beta = rho / prev_rho * alpha / omega
        p = r + beta * (p - omega * q)

        # Alpha step
        phat = M(p)  # apply preconditioner

        q = cupy.cusparse.spmv(A, phat)
        alpha = rho / cupy.vdot(r_tilde, q)
        s = r - alpha * q

        residual = cupy.linalg.norm(s)
        if residual < tol or residual < atol:
            x += alpha * phat
            info = 0
            break

        # Omega step
        shat = M(s)  # apply preconditioner

        t = cupy.cusparse.spmv(A, shat)
        omega = cupy.vdot(t, s) / cupy.square(cupy.linalg.norm(t))
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


def spilu(A, enable_boost: bool = True, tol=None, boost_val=None):
    """Compute an incomplete LU decomposition level 0
    for a sparse, square matrix (abbreviated ILU(0)).

    The resulting object is an approximation to the inverse of A.
    See https://docs.nvidia.com/cuda/cusparse/#csrilu02_solve.

    NOTE: The algorithm used by SuperLU and cuSPARSE are not
    the same. Options like `drop_tol`, `fill_factor`, and
    `diag_pivot_thresh` are not provided as they are in `scipy`'s
    version.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix
            with dimension ``(N, N)``
        enable_boost (bool): Enable numeric boost in cuSPARSE ILU(0) algorithm
        tol (float, optional): Tolerance for a numerical zero.
        (default: 1e-4)
        boost_val (float, optional): Boost value to replace a numerical zero.
            (default: 0).

    Returns:
        callable:
            A function x = M(b) that approximates a linear solution for Ax = b
        tuple:
            the info objects that need to be destroyed later

    .. seealso:: :func:`scipy.sparse.linalg.spilu`
    """
    handle = device.get_cusparse_handle()
    A_copy = A.copy()

    # support float32, float64, complex64, and complex128
    if A_copy.dtype.char in 'fdFD':
        dtype = A_copy.dtype.char
    else:
        dtype = numpy.promote_types(A_copy.dtype.char, 'f').char
    alpha = numpy.array(1, dtype).ctypes

    def A_tuple(descr):
        return (A_copy.shape[0], A_copy.nnz, descr.descriptor, A_copy.data.data.ptr,
                A_copy.indptr.data.ptr, A_copy.indices.data.ptr)

    def A_tuple_a(descr):
        return (A_copy.shape[0], A_copy.nnz, alpha.data, descr.descriptor,
                A_copy.data.data.ptr, A_copy.indptr.data.ptr, A_copy.indices.data.ptr)

    n = A.shape[0]

    # create info objects
    info_M = cusparse.createCsrilu02Info()
    info_L = cusparse.createCsrsv2Info()
    info_U = cusparse.createCsrsv2Info()

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
    descr_M.set_mat_index_base(cusparse.CUSPARSE_INDEX_BASE_ZERO)
    descr_M.set_mat_type(cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    descr_L.set_mat_index_base(cusparse.CUSPARSE_INDEX_BASE_ZERO)
    descr_L.set_mat_type(cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    descr_L.set_mat_fill_mode(cusparse.CUSPARSE_FILL_MODE_LOWER)
    descr_L.set_mat_diag_type(cusparse.CUSPARSE_DIAG_TYPE_UNIT)
    descr_U.set_mat_index_base(cusparse.CUSPARSE_INDEX_BASE_ZERO)
    descr_U.set_mat_type(cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    descr_U.set_mat_fill_mode(cusparse.CUSPARSE_FILL_MODE_UPPER)
    descr_U.set_mat_diag_type(cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT)

    # Numeric boost for csrilu02
    if enable_boost:
        enable_boost = int(enable_boost)
        tol = numpy.array(tol, numpy.float64).ctypes
        boost_val = numpy.array(boost_val, dtype).ctypes
        cupy.cusparse._call_cusparse('csrilu02_numericBoost', dtype, handle,
                                     info_M, enable_boost,
                                     tol.data, boost_val.data)

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
                                 *A_tuple(descr_M), info_M, policy_M,
                                 buff.data.ptr)
    cupy.cusparse._call_cusparse('csrsv2_analysis', dtype, handle, trans_L,
                                 *A_tuple(descr_L), info_L, policy_L,
                                 buff.data.ptr)
    cupy.cusparse._call_cusparse('csrsv2_analysis', dtype, handle, trans_U,
                                 *A_tuple(descr_U), info_U, policy_U,
                                 buff.data.ptr)

    # Perform M = L * U incomplete decomposition
    cupy.cusparse._call_cusparse('csrilu02', dtype, handle,
                                 *A_tuple(descr_M), info_M, policy_M,
                                 buff.data.ptr)

    y = cupy.empty(n, dtype=dtype)
    out = cupy.empty(n, dtype=dtype)

    def M(x):
        cupy.cusparse._call_cusparse('csrsv2_solve', dtype, handle, trans_L,
                                     *A_tuple_a(descr_L), info_L, x.data.ptr,
                                     y.data.ptr, policy_L, buff.data.ptr)
        cupy.cusparse._call_cusparse('csrsv2_solve', dtype, handle, trans_U,
                                     *A_tuple_a(descr_U), info_U, y.data.ptr,
                                     out.data.ptr, policy_U, buff.data.ptr)
        return out

    return M, (info_M, info_U, info_L)


def _destroy_ilu(info):
    if info is not None:
        cusparse.destroyCsrilu02Info(info[0])
        cusparse.destroyCsrsv2Info(info[1])
        cusparse.destroyCsrsv2Info(info[2])


def _identity(x):
    return x
