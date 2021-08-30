import numpy

import cupy
from cupy import cublas
from cupy import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system

import warnings
try:
    import scipy.sparse
    import scipy.sparse.linalg
    scipy_available = True
except ImportError:
    scipy_available = False


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
    if runtime.is_hip:
        raise RuntimeError('HIP does not support lsqr')
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    # csr_matrix is 2d
    _util._assert_stacked_square(A)
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


def lsmr(A, b, x0=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    A is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. B is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex
            matrix of the linear system. ``A`` must be
            :class:`cupy.ndarray`, :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(m,)`` or ``(m, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution. If None zeros are
            used.
        damp (float): Damping factor for regularized least-squares.
            `lsmr` solves the regularized least-squares problem
            ::

                min ||(b) - (  A   )x||
                    ||(0)   (damp*I) ||_2

            where damp is a scalar. If damp is None or 0, the system
            is solved without regularization.
        atol, btol (float):
            Stopping tolerances. `lsmr` continues iterations until a
            certain backward error estimate is smaller than some quantity
            depending on atol and btol.
        conlim (float): `lsmr` terminates if an estimate of ``cond(A)`` i.e.
            condition number of matrix exceeds `conlim`. If `conlim` is None,
            the default value is 1e+8.
        maxiter (int): Maximum number of iterations.

    Returns:
        tuple:
            - `x` (ndarray): Least-square solution returned.
            - `istop` (int): istop gives the reason for stopping::

                    0 means x=0 is a solution.

                    1 means x is an approximate solution to A*x = B,
                    according to atol and btol.

                    2 means x approximately solves the least-squares problem
                    according to atol.

                    3 means COND(A) seems to be greater than CONLIM.

                    4 is the same as 1 with atol = btol = eps (machine
                    precision)

                    5 is the same as 2 with atol = eps.

                    6 is the same as 3 with CONLIM = 1/eps.

                    7 means ITN reached maxiter before the other stopping
                    conditions were satisfied.

            - `itn` (int): Number of iterations used.
            - `normr` (float): ``norm(b-Ax)``
            - `normar` (float): ``norm(A^T (b - Ax))``
            - `norma` (float): ``norm(A)``
            - `conda` (float): Condition number of A.
            - `normx` (float): ``norm(x)``

    .. seealso:: :func:`scipy.sparse.linalg.lsmr`

    References:
        D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for
        sparse least-squares problems", SIAM J. Sci. Comput.,
        vol. 33, pp. 2950-2971, 2011.
    """
    A = _interface.aslinearoperator(A)
    b = b.squeeze()
    matvec = A.matvec
    rmatvec = A.rmatvec
    m, n = A.shape
    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim * 5

    u = b.copy()
    normb = cublas.nrm2(b)
    beta = normb.copy()
    normb = normb.get().item()
    if x0 is None:
        x = cupy.zeros((n,), dtype=A.dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensions')
        x = x0.astype(A.dtype).ravel()
        u -= matvec(x)
        beta = cublas.nrm2(u)

    beta_cpu = beta.get().item()

    v = cupy.zeros(n)
    alpha = cupy.zeros((), dtype=beta.dtype)
    alpha_cpu = 0

    if beta_cpu > 0:
        u /= beta
        v = rmatvec(u)
        alpha = cublas.nrm2(v)
        alpha_cpu = alpha.get().item()

    if alpha_cpu > 0:
        v /= alpha

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha_cpu * beta_cpu
    alphabar = alpha_cpu
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = cupy.zeros(n)
    # x = cupy.zeros(n)

    # Initialize variables for estimation of ||r||.

    betadd = beta_cpu
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha_cpu * alpha_cpu
    maxrbar = 0
    minrbar = 1e+100
    normA = alpha_cpu
    condA = 1
    normx = 0

    # Items for use in stopping rules.
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta_cpu

    # Golub-Kahan process terminates when either alpha or beta is zero.
    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha_cpu * beta_cpu
    if normar == 0:
        return x, istop, itn, normr, normar, normA, condA, normx

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u *= -alpha
        u += matvec(v)
        beta = cublas.nrm2(u)  # norm(u)
        beta_cpu = beta.get().item()

        if beta_cpu > 0:
            u /= beta
            v *= -beta
            v += rmatvec(u)
            alpha = cublas.nrm2(v)  # norm(v)
            alpha_cpu = alpha.get().item()
            if alpha_cpu > 0:
                v /= alpha

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _symOrtho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _symOrtho(alphahat, beta_cpu)
        thetanew = s * alpha_cpu
        alphabar = c * alpha_cpu

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _symOrtho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        # hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        hbar *= -(thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        # h = v - (thetanew / rho) * h
        h *= -(thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _symOrtho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = numpy.sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta_cpu * beta_cpu
        normA = numpy.sqrt(normA2)
        normA2 = normA2 + alpha_cpu * alpha_cpu

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = cublas.nrm2(x)
        normx = normx.get().item()

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = numpy.infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA*normx/normb)
        rtol = btol + atol*normA*normx/normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if istop > 0:
            break

    # The return type of SciPy is always float64. Therefore, x must be casted.
    x = x.astype(numpy.float64)

    return x, istop, itn, normr, normar, normA, condA, normx


def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False,
                       unit_diagonal=False):
    """Solves a sparse triangular system ``A x = b``.

    Args:
        A (cupyx.scipy.sparse.spmatrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray):
            Dense vector or matrix with dimension ``(M)`` or ``(M, K)``.
        lower (bool):
            Whether ``A`` is a lower or upper trinagular matrix.
            If True, it is lower triangular, otherwise, upper triangular.
        overwrite_A (bool):
            (not supported)
        overwrite_b (bool):
            Allows overwriting data in ``b``.
        unit_diagonal (bool):
            If True, diagonal elements of ``A`` are assumed to be 1 and will
            not be referencec.

    Returns:
        cupy.ndarray:
            Solution to the system ``A x = b``. The shape is the same as ``b``.
    """
    if not cusparse.check_availability('csrsm2'):
        raise NotImplementedError

    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if not isinstance(b, cupy.ndarray):
        raise TypeError('b must be cupy.ndarray')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix (A.shape: {})'.
                         format(A.shape))
    if b.ndim not in [1, 2]:
        raise ValueError('b must be 1D or 2D array (b.shape: {})'.
                         format(b.shape))
    if A.shape[0] != b.shape[0]:
        raise ValueError('The size of dimensions of A must be equal to the '
                         'size of the first dimension of b '
                         '(A.shape: {}, b.shape: {})'.format(A.shape, b.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('unsupported dtype (actual: {})'.format(A.dtype))

    if not (sparse.isspmatrix_csr(A) or sparse.isspmatrix_csc(A)):
        warnings.warn('CSR or CSC format is required. Converting to CSR '
                      'format.', sparse.SparseEfficiencyWarning)
        A = A.tocsr()
    A.sum_duplicates()

    if (overwrite_b and A.dtype == b.dtype and
            (b._c_contiguous or b._f_contiguous)):
        x = b
    else:
        x = b.astype(A.dtype, copy=True)

    cusparse.csrsm2(A, x, lower=lower, unit_diag=unit_diagonal)

    if x.dtype.char in 'fF':
        # Note: This is for compatibility with SciPy.
        dtype = numpy.promote_types(x.dtype, 'float64')
        x = x.astype(dtype)
    return x


def spsolve(A, b):
    """Solves a sparse linear system ``A x = b``

    Args:
        A (cupyx.scipy.sparse.spmatrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray):
            Dense vector or matrix with dimension ``(M)`` or ``(M, 1)``.

    Returns:
        cupy.ndarray:
            Solution to the system ``A x = b``.
    """
    if not cupy.cusolver.check_availability('csrlsvqr'):
        raise NotImplementedError
    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if not isinstance(b, cupy.ndarray):
        raise TypeError('b must be cupy.ndarray')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix (A.shape: {})'.
                         format(A.shape))
    if not (b.ndim == 1 or (b.ndim == 2 and b.shape[1] == 1)):
        raise ValueError('Invalid b.shape (b.shape: {})'.format(b.shape))
    if A.shape[0] != b.shape[0]:
        raise ValueError('matrix dimension mismatch (A.shape: {}, b.shape: {})'
                         .format(A.shape, b.shape))

    if not sparse.isspmatrix_csr(A):
        warnings.warn('CSR format is required. Converting to CSR format.',
                      sparse.SparseEfficiencyWarning)
        A = A.tocsr()
    A.sum_duplicates()
    b = b.astype(A.dtype, copy=False).ravel()

    return cupy.cusolver.csrlsvqr(A, b)


class SuperLU():

    def __init__(self, obj):
        """LU factorization of a sparse matrix.

        Args:
            obj (scipy.sparse.linalg.SuperLU): LU factorization of a sparse
                matrix, computed by `scipy.sparse.linalg.splu`, etc.
        """
        if not scipy_available:
            raise RuntimeError('scipy is not available')
        if not isinstance(obj, scipy.sparse.linalg.SuperLU):
            raise TypeError('obj must be scipy.sparse.linalg.SuperLU')

        self.shape = obj.shape
        self.nnz = obj.nnz
        self.perm_r = cupy.array(obj.perm_r)
        self.perm_c = cupy.array(obj.perm_c)
        self.L = sparse.csr_matrix(obj.L.tocsr())
        self.U = sparse.csr_matrix(obj.U.tocsr())

        self._perm_r_rev = cupy.argsort(self.perm_r)
        self._perm_c_rev = cupy.argsort(self.perm_c)

    def solve(self, rhs, trans='N'):
        """Solves linear system of equations with one or several right-hand sides.

        Args:
            rhs (cupy.ndarray): Right-hand side(s) of equation with dimension
                ``(M)`` or ``(M, K)``.
            trans (str): 'N', 'T' or 'H'.
                'N': Solves ``A * x = rhs``.
                'T': Solves ``A.T * x = rhs``.
                'H': Solves ``A.conj().T * x = rhs``.

        Returns:
            cupy.ndarray:
                Solution vector(s)
        """
        if not isinstance(rhs, cupy.ndarray):
            raise TypeError('ojb must be cupy.ndarray')
        if rhs.ndim not in (1, 2):
            raise ValueError('rhs.ndim must be 1 or 2 (actual: {})'.
                             format(rhs.ndim))
        if rhs.shape[0] != self.shape[0]:
            raise ValueError('shape mismatch (self.shape: {}, rhs.shape: {})'
                             .format(self.shape, rhs.shape))
        if trans not in ('N', 'T', 'H'):
            raise ValueError('trans must be \'N\', \'T\', or \'H\'')
        if not cusparse.check_availability('csrsm2'):
            raise NotImplementedError

        x = rhs.astype(self.L.dtype)
        if trans == 'N':
            if self.perm_r is not None:
                x = x[self._perm_r_rev]
            cusparse.csrsm2(self.L, x, lower=True, transa=trans)
            cusparse.csrsm2(self.U, x, lower=False, transa=trans)
            if self.perm_c is not None:
                x = x[self.perm_c]
        else:
            if self.perm_c is not None:
                x = x[self._perm_c_rev]
            cusparse.csrsm2(self.U, x, lower=False, transa=trans)
            cusparse.csrsm2(self.L, x, lower=True, transa=trans)
            if self.perm_r is not None:
                x = x[self.perm_r]

        if not x._f_contiguous:
            # For compatibility with SciPy
            x = x.copy(order='F')
        return x


class CusparseLU(SuperLU):

    def __init__(self, a):
        """Incomplete LU factorization of a sparse matrix.

        Args:
            a (cupyx.scipy.sparse.csr_matrix): Incomplete LU factorization of a
                sparse matrix, computed by `cusparse.csrilu02`.
        """
        if not scipy_available:
            raise RuntimeError('scipy is not available')
        if not sparse.isspmatrix_csr(a):
            raise TypeError('a must be cupyx.scipy.sparse.csr_matrix')

        self.shape = a.shape
        self.nnz = a.nnz
        self.perm_r = None
        self.perm_c = None
        # TODO(anaruse): Computes tril and triu on GPU
        a = a.get()
        al = scipy.sparse.tril(a)
        al.setdiag(1.0)
        au = scipy.sparse.triu(a)
        self.L = sparse.csr_matrix(al.tocsr())
        self.U = sparse.csr_matrix(au.tocsr())


def factorized(A):
    """Return a function for solving a sparse linear system, with A pre-factorized.

    Args:
        A (cupyx.scipy.sparse.spmatrix): Sparse matrix to factorize.

    Returns:
        callable: a function to solve the linear system of equations given in
        ``A``.

    Note:
        This function computes LU decomposition of a sparse matrix on the CPU
        using `scipy.sparse.linalg.splu`. Therefore, LU decomposition is not
        accelerated on the GPU. On the other hand, the computation of solving
        linear equations using the method returned by this function is
        performed on the GPU.

    .. seealso:: :func:`scipy.sparse.linalg.factorized`
    """
    return splu(A).solve


def splu(A, permc_spec=None, diag_pivot_thresh=None, relax=None,
         panel_size=None, options={}):
    """Computes the LU decomposition of a sparse square matrix.

    Args:
        A (cupyx.scipy.sparse.spmatrix): Sparse matrix to factorize.
        permc_spec (str): (For further augments, see
            :func:`scipy.sparse.linalg.splu`)
        diag_pivot_thresh (float):
        relax (int):
        panel_size (int):
        options (dict):

    Returns:
        cupyx.scipy.sparse.linalg.SuperLU:
            Object which has a ``solve`` method.

    Note:
        This function LU-decomposes a sparse matrix on the CPU using
        `scipy.sparse.linalg.splu`. Therefore, LU decomposition is not
        accelerated on the GPU. On the other hand, the computation of solving
        linear equations using the ``solve`` method, which this function
        returns, is performed on the GPU.

    .. seealso:: :func:`scipy.sparse.linalg.splu`
    """
    if not scipy_available:
        raise RuntimeError('scipy is not available')
    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix (A.shape: {})'
                         .format(A.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('Invalid dtype (actual: {})'.format(A.dtype))

    a = A.get().tocsc()
    a_inv = scipy.sparse.linalg.splu(
        a, permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh,
        relax=relax, panel_size=panel_size, options=options)
    return SuperLU(a_inv)


def spilu(A, drop_tol=None, fill_factor=None, drop_rule=None,
          permc_spec=None, diag_pivot_thresh=None, relax=None,
          panel_size=None, options={}):
    """Computes the incomplete LU decomposition of a sparse square matrix.

    Args:
        A (cupyx.scipy.sparse.spmatrix): Sparse matrix to factorize.
        drop_tol (float): (For further augments, see
            :func:`scipy.sparse.linalg.spilu`)
        fill_factor (float):
        drop_rule (str):
        permc_spec (str):
        diag_pivot_thresh (float):
        relax (int):
        panel_size (int):
        options (dict):

    Returns:
        cupyx.scipy.sparse.linalg.SuperLU:
            Object which has a ``solve`` method.

    Note:
        This function computes incomplete LU decomposition of a sparse matrix
        on the CPU using `scipy.sparse.linalg.spilu` (unless you set
        ``fill_factor`` to ``1``). Therefore, incomplete LU decomposition is
        not accelerated on the GPU. On the other hand, the computation of
        solving linear equations using the ``solve`` method, which this
        function returns, is performed on the GPU.

        If you set ``fill_factor`` to ``1``, this function computes incomplete
        LU decomposition on the GPU, but without fill-in or pivoting.

    .. seealso:: :func:`scipy.sparse.linalg.spilu`
    """
    if not scipy_available:
        raise RuntimeError('scipy is not available')
    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix (A.shape: {})'
                         .format(A.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('Invalid dtype (actual: {})'.format(A.dtype))

    if fill_factor == 1:
        # Computes ILU(0) on the GPU using cuSparse functions
        if not sparse.isspmatrix_csr(A):
            a = A.tocsr()
        else:
            a = A.copy()
        cusparse.csrilu02(a)
        return CusparseLU(a)

    a = A.get().tocsc()
    a_inv = scipy.sparse.linalg.spilu(
        a, fill_factor=fill_factor, drop_tol=drop_tol, drop_rule=drop_rule,
        permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh,
        relax=relax, panel_size=panel_size, options=options)
    return SuperLU(a_inv)


def _symOrtho(a, b):
    """
    A stable implementation of Givens rotation according to
    S.-C. Choi, "Iterative Methods for Singular Linear Equations
      and Least-Squares Problems", Dissertation,
      http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return numpy.sign(a), 0, abs(a)
    elif a == 0:
        return 0, numpy.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = numpy.sign(b) / numpy.sqrt(1+tau*tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = numpy.sign(a) / numpy.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def minres(A, b, x0=None, shift=0.0, tol=1e-5, maxiter=None,
           M=None, callback=None, check=False):
    """Uses MINimum RESidual iteration to solve  ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        shift (int or float): If shift != 0 then the method solves
            ``(A - shift*I)x = b``
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

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.minres`
    """
    A, M, x, b = _make_system(A, M, x0, b)

    matvec = A.matvec
    psolve = M.matvec

    n = b.shape[0]

    if maxiter is None:
        maxiter = n * 5

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = x.dtype

    eps = cupy.finfo(xtype).eps

    Ax = matvec(x)
    r1 = b - Ax
    y = psolve(r1)

    beta1 = cupy.inner(r1, y)

    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return x, 0

    beta1 = cupy.sqrt(beta1)
    beta1 = beta1.get().item()

    if check:
        # see if A is symmetric
        if not _check_symmetric(A, Ax, x, eps):
            raise ValueError('non-symmetric matrix')

        # see if M is symmetric
        if not _check_symmetric(M, y, r1, eps):
            raise ValueError('non-symmetric preconditioner')

    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = cupy.finfo(xtype).max
    cs = -1
    sn = 0
    w = cupy.zeros(n, dtype=xtype)
    w2 = cupy.zeros(n, dtype=xtype)
    r2 = r1

    while itn < maxiter:

        itn += 1
        s = 1.0 / beta
        v = s * y

        y = matvec(v)
        y -= shift * v

        if itn >= 2:
            y -= (beta / oldb) * r1

        alpha = cupy.inner(v, y)
        alpha = alpha.get().item()
        y -= (alpha / beta) * r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = cupy.inner(r2, y)
        beta = beta.get().item()
        beta = numpy.sqrt(beta)
        if beta < 0:
            raise ValueError('non-symmetric matrix')

        tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2

        if itn == 1:
            if beta / beta1 <= 10 * eps:
                istop = -1

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alpha  # delta1 = 0         deltak
        gbar = sn * dbar - cs * alpha  # gbar 1 = alfa1     gbar k
        epsln = sn * beta  # epsln2 = 0         epslnk+1
        dbar = - cs * beta  # dbar 2 = beta2     dbar k+1
        root = numpy.linalg.norm([gbar, dbar])

        # Compute the next plane rotation Qk

        gamma = numpy.linalg.norm([gbar, beta])  # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma  # ck
        sn = beta / gamma  # sk
        phi = cs * phibar  # phik
        phibar = sn * phibar  # phibark+1

        # Update  x.

        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x += phi * w

        # Go round again.

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = - epsln * z

        # Estimate various norms and test for convergence.

        Anorm = numpy.sqrt(tnorm2)
        ynorm = cupy.linalg.norm(x)
        ynorm = ynorm.get().item()
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        diag = gbar

        if diag == 0:
            diag = epsa

        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = numpy.inf
        else:
            test1 = rnorm / (Anorm * ynorm)  # ||r||  / (||A|| ||x||)
        if Anorm == 0:
            test2 = numpy.inf
        else:
            test2 = root / Anorm  # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q * H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax / gmin

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        if istop == 0:
            t1 = 1 + test1  # These tests work if tol < eps
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            # epsr = Anorm * ynorm * tol
            # if rnorm <= epsx   : istop = 2
            # if rnorm <= epsr   : istop = 1
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1

        if callback is not None:
            callback(x)

        if istop != 0:
            break

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return x, info


def _check_symmetric(op1, op2, vec, eps):
    r2 = op1 * op2
    s = cupy.inner(op2, op2)
    t = cupy.inner(vec, r2)
    z = abs(s - t)
    epsa = (s + eps) * eps ** (1.0 / 3.0)
    if z > epsa:
        return False
    return True
