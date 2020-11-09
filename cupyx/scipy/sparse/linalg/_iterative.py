import numpy

import cupy
from cupy import cublas
from cupy import cusparse
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy.sparse import csr


def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must
            be a hermitian, positive definitive matrix.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Preconditioner for
            ``A``. The preconditioner should approximate the inverse of ``A``.
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
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix (shape: {})'.format(A.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(A.dtype))
    n = A.shape[0]
    if not (b.shape == (n,) or b.shape == (n, 1)):
        raise ValueError('b has incompatible dimensins')
    b = b.astype(A.dtype).ravel()
    if n == 0:
        return cupy.empty_like(b), 0
    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return b, 0
    if atol is None:
        atol = tol * float(b_norm)
    else:
        atol = max(float(atol), tol * float(b_norm))
    if x0 is None:
        x = cupy.zeros((n,), dtype=A.dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensins')
        x = x0.astype(A.dtype).ravel()
    if maxiter is None:
        maxiter = n * 10
    matvec, psolve = _make_funcs(A, M)

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


def _make_funcs(A, M):
    matvec = _make_matvec(A)
    if M is None:
        def psolve(x): return x
    else:
        psolve = _make_matvec(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')
    return matvec, psolve


def _make_matvec(A):
    if csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        handle = device.get_cusparse_handle()
        op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        alpha = numpy.array(1.0, A.dtype)
        beta = numpy.array(0.0, A.dtype)
        cuda_dtype = cusparse._dtype_to_DataType(A.dtype)
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
    else:
        def matvec(x): return A @ x
    return matvec
