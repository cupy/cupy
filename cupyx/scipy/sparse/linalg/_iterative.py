import cupy


def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None,
       atol=None):
    """Uses Conjugate Gradient iteration to solve ``Ax = b``.

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The real or complex
            matrix of the linear system with shape ``(n, n)``. ``A`` must
            be a hermitian, positive definitive matrix.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): Preconditioner for
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
    if atol is None:
        if b_norm == 0:
            atol = tol
        else:
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

    def matvec(x): return A @ x

    if M is None:
        def psolve(x): return x
    else:
        def psolve(x): return M @ x
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    r = b - matvec(x)
    iters = 0
    rho = 0
    while iters < maxiter:
        z = psolve(r)
        rho1 = rho
        rho = cupy.dot(r.conj(), z)
        if iters == 0:
            p = z
        else:
            beta = rho / rho1
            p = z + beta * p
        q = matvec(p)
        alpha = rho / cupy.dot(p.conj(), q)
        x = x + alpha * p
        r = r - alpha * q
        iters += 1
        if callback is not None:
            callback(x)
        resid = cupy.linalg.norm(r)
        if resid <= atol:
            break

    info = 0
    if iters == maxiter and not (resid <= atol):
        info = iters

    return x, info
