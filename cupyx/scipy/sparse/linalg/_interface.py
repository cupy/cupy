import warnings

import cupy

from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util


class LinearOperator(object):
    """LinearOperator(shape, matvec, rmatvec=None, matmat=None, dtype=None, \
rmatmat=None)

    Common interface for performing matrix vector products

    To construct a concrete LinearOperator, either pass appropriate callables
    to the constructor of this class, or subclass it.

    Args:
        shape (tuple):  Matrix dimensions ``(M, N)``.
        matvec (callable f(v)):  Returns returns ``A * v``.
        rmatvec (callable f(v)):  Returns ``A^H * v``, where ``A^H`` is the
                                  conjugate transpose of ``A``.
        matmat (callable f(V)):  Returns ``A * V``, where ``V`` is a dense
                                 matrix with dimensions ``(N, K)``.
        dtype (dtype):  Data type of the matrix.
        rmatmat (callable f(V)):  Returns ``A^H * V``, where ``V`` is a dense
                                  matrix with dimensions ``(M, K)``.

    .. seealso:: :class:`scipy.sparse.linalg.LinearOperator`
    """

    ndim = 2

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return super(LinearOperator, cls).__new__(_CustomLinearOperator)
        else:
            obj = super(LinearOperator, cls).__new__(cls)

            if (type(obj)._matvec == LinearOperator._matvec
                    and type(obj)._matmat == LinearOperator._matmat):
                warnings.warn('LinearOperator subclass should implement'
                              ' at least one of _matvec and _matmat.',
                              category=RuntimeWarning, stacklevel=2)

            return obj

    def __init__(self, dtype, shape):
        """Initialize this :class:`LinearOperator`
        """
        if dtype is not None:
            dtype = cupy.dtype(dtype)

        shape = tuple(shape)
        if not _util.isshape(shape):
            raise ValueError('invalid shape %r (must be 2-d)' % (shape,))

        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Called from subclasses at the end of the `__init__` routine.
        """
        if self.dtype is None:
            v = cupy.zeros(self.shape[-1])
            self.dtype = self.matvec(v).dtype

    def _matmat(self, X):
        """Default matrix-matrix multiplication handler.
        """

        return cupy.hstack([self.matvec(col.reshape(-1, 1)) for col in X.T])

    def _matvec(self, x):
        """Default matrix-vector multiplication handler.
        """
        return self.matmat(x.reshape(-1, 1))

    def matvec(self, x):
        """Matrix-vector multiplication.
        """

        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.
        """

        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError(
                'invalid shape returned by user-defined rmatvec()')

        return y

    def _rmatvec(self, x):
        """Default implementation of _rmatvec; defers to adjoint.
        """
        if type(self)._adjoint == LinearOperator._adjoint:
            # _adjoint not overridden, prevent infinite recursion
            raise NotImplementedError
        else:
            return self.H.matvec(x)

    def matmat(self, X):
        """Matrix-matrix multiplication.
        """

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[1]:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._matmat(X)

        return Y

    def rmatmat(self, X):
        """Adjoint matrix-matrix multiplication.
        """

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[0]:
            raise ValueError('dimension mismatch: %r, %r'
                             % (self.shape, X.shape))

        Y = self._rmatmat(X)
        return Y

    def _rmatmat(self, X):
        """Default implementation of _rmatmat defers to rmatvec or adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            return cupy.hstack([self.rmatvec(col.reshape(-1, 1))
                                for col in X.T])
        else:
            return self.H.matmat(X)

    def __call__(self, x):
        return self*x

    def __mul__(self, x):
        return self.dot(x)

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif cupy.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError('expected 1-d or 2-d array, got %r'
                                 % x)

    def __matmul__(self, other):
        if cupy.isscalar(other):
            raise ValueError('Scalar operands are not allowed, '
                             'use \'*\' instead')
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if cupy.isscalar(other):
            raise ValueError('Scalar operands are not allowed, '
                             'use \'*\' instead')
        return self.__rmul__(other)

    def __rmul__(self, x):
        if cupy.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __pow__(self, p):
        if cupy.isscalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M, N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def adjoint(self):
        """Hermitian adjoint.
        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """Transpose this linear operator.
        """
        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """Default implementation of _adjoint; defers to rmatvec."""
        return _AdjointLinearOperator(self)

    def _transpose(self):
        """ Default implementation of _transpose; defers to rmatvec + conj"""
        return _TransposedLinearOperator(self)


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, rmatmat=None):
        super(_CustomLinearOperator, self).__init__(dtype, shape)

        self.args = ()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl = matmat

        self._init_dtype()

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._matmat(X)

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _rmatvec(self, x):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError('rmatvec is not defined')
        return self.__rmatvec_impl(x)

    def _rmatmat(self, X):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(X)
        else:
            return super(_CustomLinearOperator, self)._rmatmat(X)

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     matmat=self.__rmatmat_impl,
                                     rmatmat=self.__matmat_impl,
                                     dtype=self.dtype)


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super(_AdjointLinearOperator, self).__init__(
            dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        return self.A._rmatvec(x)

    def _rmatvec(self, x):
        return self.A._matvec(x)

    def _matmat(self, x):
        return self.A._rmatmat(x)

    def _rmatmat(self, x):
        return self.A._matmat(x)


class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super(_TransposedLinearOperator, self).__init__(
            dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        # NB. cupy.conj works also on sparse matrices
        return cupy.conj(self.A._rmatvec(cupy.conj(x)))

    def _rmatvec(self, x):
        return cupy.conj(self.A._matvec(cupy.conj(x)))

    def _matmat(self, x):
        # NB. cupy.conj works also on sparse matrices
        return cupy.conj(self.A._rmatmat(cupy.conj(x)))

    def _rmatmat(self, x):
        return cupy.conj(self.A._matmat(cupy.conj(x)))


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)
    return cupy.find_common_type(dtypes, [])


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape != B.shape:
            raise ValueError('cannot add %r and %r: shape mismatch'
                             % (A, B))
        self.args = (A, B)
        super(_SumLinearOperator, self).__init__(_get_dtype([A, B]), A.shape)

    def _matvec(self, x):
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    def _rmatvec(self, x):
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    def _rmatmat(self, x):
        return self.args[0].rmatmat(x) + self.args[1].rmatmat(x)

    def _matmat(self, x):
        return self.args[0].matmat(x) + self.args[1].matmat(x)

    def _adjoint(self):
        A, B = self.args
        return A.H + B.H


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise ValueError('cannot multiply %r and %r: shape mismatch'
                             % (A, B))
        super(_ProductLinearOperator, self).__init__(_get_dtype([A, B]),
                                                     (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _rmatmat(self, x):
        return self.args[1].rmatmat(self.args[0].rmatmat(x))

    def _matmat(self, x):
        return self.args[0].matmat(self.args[1].matmat(x))

    def _adjoint(self):
        A, B = self.args
        return B.H * A.H


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if not cupy.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        dtype = _get_dtype([A], [type(alpha)])
        super(_ScaledLinearOperator, self).__init__(dtype, A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return cupy.conj(self.args[1]) * self.args[0].rmatvec(x)

    def _rmatmat(self, x):
        return cupy.conj(self.args[1]) * self.args[0].rmatmat(x)

    def _matmat(self, x):
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self):
        A, alpha = self.args
        return A.H * cupy.conj(alpha)


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if A.shape[0] != A.shape[1]:
            raise ValueError('square LinearOperator expected, got %r' % A)
        if not _util.isintlike(p) or p < 0:
            raise ValueError('non-negative integer expected as p')

        super(_PowerLinearOperator, self).__init__(_get_dtype([A]), A.shape)
        self.args = (A, p)

    def _power(self, fun, x):
        res = cupy.array(x, copy=True)
        for i in range(self.args[1]):
            res = fun(res)
        return res

    def _matvec(self, x):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x):
        return self._power(self.args[0].rmatvec, x)

    def _rmatmat(self, x):
        return self._power(self.args[0].rmatmat, x)

    def _matmat(self, x):
        return self._power(self.args[0].matmat, x)

    def _adjoint(self):
        A, p = self.args
        return A.H ** p


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self)
        return self.__adj


class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint):
        self.A = adjoint.A.T.conj()
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = adjoint.shape[1], adjoint.shape[0]

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint


class IdentityOperator(LinearOperator):
    def __init__(self, shape, dtype=None):
        super(IdentityOperator, self).__init__(dtype, shape)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _rmatmat(self, x):
        return x

    def _matmat(self, x):
        return x

    def _adjoint(self):
        return self


def aslinearoperator(A):
    """Return `A` as a LinearOperator.

    Args:
        A (array-like):
            The input array to be converted to a `LinearOperator` object.
            It may be any of the following types:

               * :class:`cupy.ndarray`
               * sparse matrix (e.g. ``csr_matrix``, ``coo_matrix``, etc.)
               * :class:`cupyx.scipy.sparse.linalg.LinearOperator`
               * object with ``.shape`` and ``.matvec`` attributes

    Returns:
        cupyx.scipy.sparse.linalg.LinearOperator: `LinearOperator` object

    .. seealso:: :func:`scipy.sparse.aslinearoperator``
    """
    if isinstance(A, LinearOperator):
        return A

    elif isinstance(A, cupy.ndarray):
        if A.ndim > 2:
            raise ValueError('array must have ndim <= 2')
        A = cupy.atleast_2d(A)
        return MatrixLinearOperator(A)

    elif sparse.isspmatrix(A):
        return MatrixLinearOperator(A)

    else:
        if hasattr(A, 'shape') and hasattr(A, 'matvec'):
            rmatvec = None
            rmatmat = None
            dtype = None

            if hasattr(A, 'rmatvec'):
                rmatvec = A.rmatvec
            if hasattr(A, 'rmatmat'):
                rmatmat = A.rmatmat
            if hasattr(A, 'dtype'):
                dtype = A.dtype
            return LinearOperator(A.shape, A.matvec, rmatvec=rmatvec,
                                  rmatmat=rmatmat, dtype=dtype)

        else:
            raise TypeError('type not understood')
