import numpy
import six

import cupy
from cupy.core import core
from cupy.sparse import util


class spmatrix(object):

    """Base class of all sparse matrixes.

    See :class:`scipy.sparse.spmatrix`
    """

    __array_priority__ = 101

    def __init__(self, maxprint=50):
        if self.__class__ == spmatrix:
            raise ValueError(
                'This class is not intended to be instantiated directly.')
        self.maxprint = maxprint

    @property
    def device(self):
        """CUDA device on which this array resides."""
        raise NotImplementedError

    def get(self, stream=None):
        """Return a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.spmatrix: An array on host memory.

        """
        raise NotImplementedError

    def __len__(self):
        raise TypeError('sparse matrix length is ambiguous; '
                        'use getnnz() or shape[0]')

    def __str__(self):
        # TODO(unno): Do not use get method which is only available when scipy
        # is installed.
        return str(self.get())

    def __iter__(self):
        for r in six.moves.range(self.shape[0]):
            yield self[r, :]

    def __bool__(self):
        if self.shape == (1, 1):
            return self.nnz != 0
        else:
            raise ValueError('The truth value of an array with more than one '
                             'element is ambiguous. Use a.any() or a.all().')

    __nonzero__ = __bool__

    def __eq__(self, other):
        return self.tocsr().__eq__(other)

    def __ne__(self, other):
        return self.tocsr().__ne__(other)

    def __lt__(self, other):
        return self.tocsr().__lt__(other)

    def __gt__(self, other):
        return self.tocsr().__gt__(other)

    def __le__(self, other):
        return self.tocsr().__le__(other)

    def __ge__(self, other):
        return self.tocsr().__ge__(other)

    def __abs__(self):
        return self.tocsr().__abs__()

    def __add__(self, other):
        return self.tocsr().__add__(other)

    def __radd__(self, other):
        return self.tocsr().__radd__(other)

    def __sub__(self, other):
        return self.tocsr().__sub__(other)

    def __rsub__(self, other):
        return self.tocsr().__rsub__(other)

    def __mul__(self, other):
        return self.tocsr().__mul__(other)

    def __rmul__(self, other):
        if cupy.isscalar(other) or isdense(other) and other.ndim == 0:
            return self * other
        else:
            try:
                tr = other.T
            except AttributeError:
                return NotImplemented
            return (self.T * tr).T

    def __div__(self, other):
        return self.tocsr().__div__(other)

    def __rdiv__(self, other):
        return self.tocsr().__rdiv__(other)

    def __truediv__(self, other):
        return self.tocsr().__truediv__(other)

    def __rtruediv__(self, other):
        return self.tocsr().__rdtrueiv__(other)

    def __neg__(self):
        return -self.tocsr()

    def __iadd__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        return NotImplemented

    def __pow__(self, other):
        """Calculates n-th power of the matrix.

        This method calculates n-th power of a given matrix. The matrix must
        be a squared matrix, and a given exponent must be an integer.

        Args:
            other (int): Exponent.

        Returns:
            cupy.sparse.spmatrix: A sparse matrix representing n-th power of
                this matrix.

        """
        m, n = self.shape
        if m != n:
            raise TypeError('matrix is not square')

        if util.isintlike(other):
            other = int(other)
            if other < 0:
                raise ValueError('exponent must be >= 0')

            if other == 0:
                return cupy.sparse.identity(m, dtype=self.dtype, format='csr')
            elif other == 1:
                return self.copy()
            else:
                tmp = self.__pow__(other // 2)
                if other % 2:
                    return self * tmp * tmp
                else:
                    return tmp * tmp
        elif util.isscalarlike(other):
            raise ValueError('exponent must be an integer')
        else:
            return NotImplemented

    @property
    def A(self):
        """Dense ndarray representation of this matrix.

        This property is equivalent to :meth:`~cupy.sparse.spmatrix.toarray`
        method.

        """
        return self.toarray()

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return self.getH()

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self.getnnz()

    @property
    def nnz(self):
        return self.getnnz()

    @property
    def shape(self):
        return self.get_shape()

    @shape.setter
    def shape(self, value):
        self.set_shape(value)

    def asformat(self, format):
        """Return this matrix in a given sparse format.

        Args:
            format (str or None): Format you need.
        """
        if format is None or format == self.format:
            return self
        else:
            return getattr(self, 'to' + format)()

    def asfptype(self):
        """Upcasts matrix to a floating point format.

        When the matrix has floating point type, the method returns itself.
        Otherwise it makes a copy with floating point type and the same format.

        Returns:
            cupy.sparse.spmatrix: A matrix with float type.

        """
        if self.dtype.kind == 'f':
            return self
        else:
            typ = numpy.result_type(self.dtype, 'f')
            return self.astype(typ)

    def astype(self, t):
        """Casts the array to given data type.

        Args:
            t: Type specifier.

        Returns:
            cupy.sparse.spmatrix:
                A copy of the array with the given type and the same format.

        """
        return self.tocsr().astype(t).asformat(self.format)

    def conj(self):
        return self.tocsr().conj()

    def conjugate(self):
        return self.conj()

    def copy(self):
        """Returns a copy of this matrix."""
        return self.__class__(self, copy=True)

    def count_nonzero(self):
        """Number of non-zero entries, equivalent to"""
        raise NotImplementedError

    def diagonal(self):
        """Returns the main diagonal of the matrix"""
        return self.tocsr().diagonal()

    def dot(self, other):
        """Ordinary dot product"""
        return self * other

    def getH(self):
        return self.transpose().conj()

    def get_shape(self):
        raise NotImplementedError

    # TODO(unno): Implement getcol

    def getformat(self):
        return self.format

    def getmaxprint(self):
        return self.maxprint

    def getnnz(self, axis=None):
        """Number of stored values, including explicit zeros."""
        raise NotImplementedError

    # TODO(unno): Implement getrow

    def maximum(self, other):
        return self.tocsr().maximum(other)

    # TODO(unno): Implement mean

    def minimum(self, other):
        return self.tocsr().minimum(other)

    def multiply(self, other):
        """Point-wise multiplication by another matrix"""
        return self.tocsr().multiply(other)

    # TODO(unno): Implement nonzero

    def power(self, n, dtype=None):
        return self.tocsr().power(n, dtype=dtype)

    def reshape(self, shape, order='C'):
        """Gives a new shape to a sparse matrix without changing its data."""
        raise NotImplementedError

    def set_shape(self, shape):
        self.reshape(shape)

    # TODO(unno): Implement setdiag

    # TODO(unno): Implement sum

    def toarray(self, order=None, out=None):
        """Return a dense ndarray representation of this matrix."""
        return self.tocsr().toarray(order=order, out=out)

    def tobsr(self, blocksize=None, copy=False):
        """Convert this matrix to Block Sparse Row format."""
        return self.tocsr(copy=copy).tobsr(copy=False)

    def tocoo(self, copy=False):
        """Convert this matrix to COOrdinate format."""
        return self.tocsr(copy=copy).tocoo(copy=False)

    def tocsc(self, copy=False):
        """Convert this matrix to Compressed Sparse Column format."""
        return self.tocsr(copy=copy).tocsc(copy=False)

    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format."""
        raise NotImplementedError

    def todense(self, order=None, out=None):
        """Return a dense matrix representation of this matrix."""
        return self.toarray(order=order, out=out)

    def todia(self, copy=False):
        """Convert this matrix to sparse DIAgonal format."""
        return self.tocsr(copy=copy).todia(copy=False)

    def todok(self, copy=False):
        """Convert this matrix to Dictionary Of Keys format."""
        return self.tocsr(copy=copy).todok(copy=False)

    def tolil(self, copy=False):
        """Convert this matrix to LInked List format."""
        return self.tocsr(copy=copy).tolil(copy=False)

    def transpose(self, axes=None, copy=False):
        """Reverses the dimensions of the sparse matrix."""
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False)


def issparse(x):
    """Checks if a given matrix is a sparse matrix.

    Returns:
        bool: Returns if ``x`` is :class:`cupy.sparse.spmatrix` that is a base
            class of all sparse matrix classes.

    """
    return isinstance(x, spmatrix)


isspmatrix = issparse


def isdense(x):
    return isinstance(x, core.ndarray)
