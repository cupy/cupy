import numpy

import cupy
from cupy import cublas
from cupy import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import csr
from cupyx.scipy.sparse.linalg import _interface


def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``. ``A`` must be a hermitian,
            positive definitive matrix with type of :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.cg`
    """
    A, M, x, b = _make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec

    n = A.shape[0]
    if maxiter is None:
        maxiter = n * 10
    if n == 0:
        return cupy.empty_like(b), 0
    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return b, 0
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))

    r = b - matvec(x)
    iters = 0
    rho = 0
    while iters < maxiter:
        z = psolve(r)
        rho1 = rho
        rho = cublas.dotc(r, z)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            p = z + beta * p
        q = matvec(p)
        alpha = rho / cublas.dotc(p, q)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        resid = cublas.nrm2(r)
        if resid <= atol:
            break

    info = 0
    if iters == maxiter and not (resid <= atol):
        info = iters

    return x, info


def gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None,
          callback=None, atol=None, callback_type=None):
    """Uses Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        restart (int): Number of iterations between restarts. Larger values
            increase iteration cost, but may be necessary for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call on every restart.
            It is called as ``callback(arg)``, where ``arg`` is selected by
            ``callback_type``.
        callback_type (str): 'x' or 'pr_norm'. If 'x', the current solution
            vector is used as an argument of callback function. if 'pr_norm',
            relative (preconditioned) residual norm is used as an arugment.
        atol (float): Tolerance for convergence.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: :func:`scipy.sparse.linalg.gmres`
    """
    A, M, x, b = _make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec

    n = A.shape[0]
    if n == 0:
        return cupy.empty_like(b), 0
    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return b, 0
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 20
    restart = min(restart, n)
    if callback_type is None:
        callback_type = 'pr_norm'
    if callback_type not in ('x', 'pr_norm'):
        raise ValueError('Unknown callback_type: {}'.format(callback_type))
    if callback is None:
        callback_type = None

    V = cupy.empty((n, restart), dtype=A.dtype, order='F')
    H = cupy.zeros((restart+1, restart), dtype=A.dtype, order='F')
    e = numpy.zeros((restart+1,), dtype=A.dtype)

    compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        mx = psolve(x)
        r = b - matvec(mx)
        r_norm = cublas.nrm2(r)
        if callback_type == 'x':
            callback(mx)
        elif callback_type == 'pr_norm' and iters > 0:
            callback(r_norm / b_norm)
        if r_norm <= atol or iters >= maxiter:
            break
        v = r / r_norm
        V[:, 0] = v
        e[0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            z = psolve(v)
            u = matvec(z)
            H[:j+1, j], u = compute_hu(u, j)
            cublas.nrm2(u, out=H[j+1, j])
            if j+1 < restart:
                v = u / H[j+1, j]
                V[:, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if tha matrix size is small.
        ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        y = cupy.array(ret[0])
        x += V @ y
        iters += restart

    info = 0
    if iters == maxiter and not (r_norm <= atol):
        info = iters
    return mx, info


def _make_system(A, M, x0, b):
    """Make a linear system Ax = b

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix or
            cupyx.scipy.sparse.LinearOperator): sparse or dense matrix.
        M (cupy.ndarray or cupyx.scipy.sparse.spmatrix or
            cupyx.scipy.sparse.LinearOperator): preconditioner.
        x0 (cupy.ndarray): initial guess to iterative method.
        b (cupy.ndarray): right hand side.

    Returns:
        tuple:
            It returns (A, M, x, b).
            A (LinaerOperator): matrix of linear system
            M (LinearOperator): preconditioner
            x (cupy.ndarray): initial guess
            b (cupy.ndarray): right hand side.
    """
    fast_matvec = _make_fast_matvec(A)
    A = _interface.aslinearoperator(A)
    if fast_matvec is not None:
        A = _interface.LinearOperator(A.shape, matvec=fast_matvec,
                                      rmatvec=A.rmatvec, dtype=A.dtype)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix (shape: {})'.format(A.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(A.dtype))
    n = A.shape[0]
    if not (b.shape == (n,) or b.shape == (n, 1)):
        raise ValueError('b has incompatible dimensions')
    b = b.astype(A.dtype).ravel()
    if x0 is None:
        x = cupy.zeros((n,), dtype=A.dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensions')
        x = x0.astype(A.dtype).ravel()
    if M is None:
        M = _interface.IdentityOperator(shape=A.shape, dtype=A.dtype)
    else:
        fast_matvec = _make_fast_matvec(M)
        M = _interface.aslinearoperator(M)
        if fast_matvec is not None:
            M = _interface.LinearOperator(M.shape, matvec=fast_matvec,
                                          rmatvec=M.rmatvec, dtype=M.dtype)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')
    return A, M, x, b


def _make_fast_matvec(A):
    matvec = None
    if csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        handle = device.get_cusparse_handle()
        op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        alpha = numpy.array(1.0, A.dtype)
        beta = numpy.array(0.0, A.dtype)
        cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
        x = cupy.empty((A.shape[0],), dtype=A.dtype)
        y = cupy.empty((A.shape[0],), dtype=A.dtype)
        desc_A = cusparse.SpMatDescriptor.create(A)
        desc_x = cusparse.DnVecDescriptor.create(x)
        desc_y = cusparse.DnVecDescriptor.create(y)
        buff_size = _cusparse.spMV_bufferSize(
            handle, op_a, alpha.ctypes.data, desc_A.desc, desc_x.desc,
            beta.ctypes.data, desc_y.desc, cuda_dtype, alg)
        buff = cupy.empty(buff_size, cupy.int8)
        del x, desc_x, y, desc_y

        def matvec(x):
            y = cupy.empty_like(x)
            desc_x = cusparse.DnVecDescriptor.create(x)
            desc_y = cusparse.DnVecDescriptor.create(y)
            _cusparse.spMV(
                handle, op_a, alpha.ctypes.data, desc_A.desc, desc_x.desc,
                beta.ctypes.data, desc_y.desc, cuda_dtype, alg, buff.data.ptr)
            return y

    return matvec


def _make_compute_hu(V):
    handle = device.get_cublas_handle()
    if V.dtype.char == 'f':
        gemv = _cublas.sgemv
    elif V.dtype.char == 'd':
        gemv = _cublas.dgemv
    elif V.dtype.char == 'F':
        gemv = _cublas.cgemv
    elif V.dtype.char == 'D':
        gemv = _cublas.zgemv
    n = V.shape[0]
    one = numpy.array(1.0, V.dtype)
    zero = numpy.array(0.0, V.dtype)
    mone = numpy.array(-1.0, V.dtype)

    def compute_hu(u, j):
        # h = V[:, :j+1].conj().T @ u
        # u -= V[:, :j+1] @ h
        h = cupy.empty((j+1,), dtype=V.dtype)
        gemv(handle, _cublas.CUBLAS_OP_C, n, j+1, one.ctypes.data, V.data.ptr,
             n, u.data.ptr, 1, zero.ctypes.data, h.data.ptr, 1)
        gemv(handle, _cublas.CUBLAS_OP_N, n, j+1, mone.ctypes.data, V.data.ptr,
             n, h.data.ptr, 1, one.ctypes.data, u.data.ptr, 1)
        return h, u
    return compute_hu
