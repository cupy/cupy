import numpy

import cupy
from cupy import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.linalg import _util
from cupyx.scipy import sparse

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
