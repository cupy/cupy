from __future__ import annotations

import numbers
import warnings

import numpy

import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils


try:
    import scipy.sparse as _sparse
    SparseWarning = _sparse.SparseWarning
    SparseEfficiencyWarning = _sparse.SparseEfficiencyWarning
except ImportError:
    class SparseWarning(Warning):  # type: ignore
        pass

    class SparseEfficiencyWarning(SparseWarning):  # type: ignore
        pass


# Format -> human-readable name (mirrors scipy.sparse._base._formats).
_format_names = {
    'csr': 'Compressed Sparse Row',
    'csc': 'Compressed Sparse Column',
    'coo': 'COOrdinate',
    'dia': 'DIAgonal',
    'dok': 'Dictionary Of Keys',
    'lil': 'List of Lists',
    'bsr': 'Block Sparse Row',
}


class _spbase:
    """Common base class for all sparse arrays and matrices.

    .. seealso:: :class:`scipy.sparse._base._spbase`
    """

    __array_priority__ = 101
    # Class default since ``__init__`` chains across format subclasses
    # don't always reach ``spmatrix.__init__``.
    maxprint = 50
    # Accepted dimensionalities (mirrors scipy's ``_allow_nd``).  All
    # matrices and the base default are 2-D only; ``coo_array`` and
    # ``csr_array`` override this to also accept 1-D.
    _allow_nd: tuple[int, ...] = (2,)

    def __class_getitem__(cls, args):
        # ``coo_array[int]``-style typing aliases (scipy 1.16+).
        import types
        return types.GenericAlias(cls, args)

    @property
    def device(self):
        """CUDA device on which this object resides."""
        raise NotImplementedError

    def get(self, stream=None):
        """Return a copy of this object on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given,
                the copy runs asynchronously. Otherwise, the copy is
                synchronous.

        Returns:
            scipy.sparse: A SciPy sparse object on host memory of the
            matching format and array/matrix type.

        """
        raise NotImplementedError

    def __len__(self):
        raise TypeError('sparse array length is ambiguous; '
                        'use shape[0] or .nnz')

    def __repr__(self):
        format_name = _format_names.get(self.format, self.format)
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        return (
            f"<{format_name} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements and shape {self.shape}>")

    def __str__(self):
        # Delegate to scipy via ``self.get()`` so the output (including
        # version-specific quirks like DIA's "(N diagonals)" annotation)
        # tracks the installed scipy.  Fall back to ``repr`` only when
        # ``get()`` is unavailable.
        try:
            return str(self.get())
        except (RuntimeError, NotImplementedError):
            return repr(self)

    def __iter__(self):
        if self.ndim == 1:
            # Iterate elements (0-D results), like cupy dense 1-D arrays.
            for i in range(self.shape[0]):
                yield self[i]
        else:
            for r in range(self.shape[0]):
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

    def __neg__(self):
        return -self.tocsr()

    def __add__(self, other):
        return self.tocsr().__add__(other)

    def __radd__(self, other):
        return self.tocsr().__radd__(other)

    def __sub__(self, other):
        return self.tocsr().__sub__(other)

    def __rsub__(self, other):
        return self.tocsr().__rsub__(other)

    # Array semantics: * is element-wise (spmatrix overrides to matmul)
    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __div__(self, other):
        return self.tocsr().__div__(other)

    def __rdiv__(self, other):
        return self.tocsr().__rdiv__(other)

    def __truediv__(self, other):
        return self.tocsr().__truediv__(other)

    def __rtruediv__(self, other):
        return self.tocsr().__rtruediv__(other)

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

    # Array semantics: ** is element-wise (spmatrix overrides to matrix power)
    def __pow__(self, other):
        return self.power(other)

    # matmul (@) operator
    def __matmul__(self, other):
        if _util.isscalarlike(other):
            raise ValueError('Scalar operands are not allowed, '
                             'use \'*\' instead')
        return self._matmul_dispatch(other)

    def __rmatmul__(self, other):
        if _util.isscalarlike(other):
            raise ValueError('Scalar operands are not allowed, '
                             'use \'*\' instead')
        return self._rmatmul_dispatch(other)

    def _matmul_dispatch(self, other):
        """Default: convert to CSR.  Format subclasses override."""
        return self.tocsr()._matmul_dispatch(other)

    def _rmatmul_dispatch(self, other):
        if cupy.isscalar(other) or (isdense(other) and other.ndim == 0):
            return self._matmul_dispatch(other)
        else:
            try:
                tr = other.T
            except AttributeError:
                return NotImplemented
            return (self.T._matmul_dispatch(tr)).T

    @property
    def T(self):
        return self.transpose()

    @property
    def mT(self):
        """Matrix transpose.

        Equivalent to :func:`cupyx.scipy.sparse.matrix_transpose`.
        CuPy sparse types are 2-D only.
        """
        n = self.ndim
        if n < 2:
            raise ValueError(
                'Array must be at least 2-dimensional, '
                f'but it is {n}-D')
        return self.transpose()

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return self._getnnz()

    @property
    def nnz(self):
        return self._getnnz()

    @property
    def shape(self):
        return self._shape

    @property
    def _shape_as_2d(self):
        s = self._shape
        return (1, s[-1]) if len(s) == 1 else s

    def _require_2d(self, op):
        """Raise ``ValueError`` if this is not a 2-D array/matrix.

        Central guard for operations that are only meaningful in 2-D
        (e.g. ``diagonal``/``setdiag``); keeps the message consistent
        and makes it obvious which ops still assume 2-D.
        """
        if self.ndim != 2:
            raise ValueError(f'{op} requires two dimensions')

    def _squeeze_to_1d(self, result):
        """Squeeze a 2-D ``(1, N)`` op result back to 1-D.

        The 1-D sparse-array ops run on the ``(1, N)`` backing (see
        :meth:`_as_2d`) and pass the result here.  When the other operand
        is itself 2-D it can broadcast the result up to ``(K, N)`` with
        ``K > 1`` (e.g. ``multiply``/``maximum`` against a 2-D operand);
        such a result is genuinely 2-D and is returned unchanged, matching
        scipy.  A true ``(1, N)`` result is collapsed to 1-D: a CSR result
        is rewrapped as a 1-D CSR (preserving format, matching scipy); any
        other sparse result is reshaped to 1-D; a dense result is raveled.
        """
        if issparse(result):
            m, n = result.shape
            if m != 1:
                # A 2-D operand broadcast the result up; keep it 2-D.
                return result
            if result.format == 'csr':
                return self._csr_container._from_parts(
                    result.data, result.indices, result.indptr, (n,),
                    has_canonical_format=getattr(
                        result, '_has_canonical_format', None),
                    has_sorted_indices=getattr(
                        result, '_has_sorted_indices', None))
            return result.reshape((n,))
        if result.ndim == 2 and result.shape[0] != 1:
            # Dense result broadcast up by a 2-D operand -- genuinely 2-D.
            return result
        return result.reshape(-1)

    # Container properties: default to array types.
    # spmatrix overrides these to return matrix types.

    @property
    def _csr_container(self):
        from cupyx.scipy.sparse._csr import csr_array
        return csr_array

    @property
    def _csc_container(self):
        from cupyx.scipy.sparse._csc import csc_array
        return csc_array

    @property
    def _coo_container(self):
        from cupyx.scipy.sparse._coo import coo_array
        return coo_array

    @property
    def _dia_container(self):
        from cupyx.scipy.sparse._dia import dia_array
        return dia_array

    def _get_index_dtype(self, arrays=(), maxval=None, check_contents=False):
        """Wraps get_index_dtype: arrays never downcast (check_contents
        is disabled for sparray instances).
        """
        return _sputils.get_index_dtype(
            arrays, maxval,
            check_contents and not isinstance(self, sparray))

    def asformat(self, format):
        """Return this matrix in a given sparse format.

        Args:
            format (str or None): Format you need.
        """
        if format is None or format == self.format:
            return self
        else:
            return getattr(self, 'to' + format)()

    def astype(self, dtype, copy=True):
        """Cast the array elements to a specified type.

        Args:
            dtype: Target dtype.
            copy (bool): If ``True`` (default), the returned array does
                not share memory with ``self``.  If ``False``, ``self``
                is returned unchanged when the dtype already matches.

        Returns:
            cupyx.scipy.sparse: Sparse object with the requested dtype
            and the same format as ``self``.
        """
        dtype = numpy.dtype(dtype)
        if self.dtype != dtype:
            return self.tocsr().astype(
                dtype, copy=copy).asformat(self.format)
        if copy:
            return self.copy()
        return self

    def conj(self, copy=True):
        """Element-wise complex conjugation.

        If the matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Args:
            copy (bool):
                If True, the result is guaranteed to not share data with self.

        Returns:
            cupyx.scipy.sparse.spmatrix : The element-wise complex conjugate.

        """
        if self.dtype.kind == 'c':
            return self.tocsr(copy=copy).conj(copy=False)
        elif copy:
            return self.copy()
        else:
            return self

    def conjugate(self, copy=True):
        return self.conj(copy=copy)

    conjugate.__doc__ = conj.__doc__

    def copy(self):
        """Returns a copy of this matrix.

        No data/indices will be shared between the returned value and current
        matrix.
        """
        return self.__class__(self, copy=True)

    def count_nonzero(self, axis=None):
        """Number of non-zero entries.

        Subclasses override this; ``_spbase`` itself does not implement
        a generic counter.
        """
        raise NotImplementedError

    def diagonal(self, k=0):
        """Returns the k-th diagonal of the matrix.

        Args:
            k (int, optional): Which diagonal to get, corresponding to elements
            a[i, i+k]. Default: 0 (the main diagonal).

        Returns:
            cupy.ndarray : The k-th diagonal.
        """
        return self.tocsr().diagonal(k=k)

    def trace(self, offset=0):
        """Returns the sum along diagonals of the sparse matrix.

        Args:
            offset (int): Which diagonal to get. Default: 0.

        Returns:
            cupy.ndarray: Sum along diagonal.

        .. seealso::
           :meth:`scipy.sparse.spmatrix.trace`
        """
        return self.diagonal(k=offset).sum()

    def dot(self, other):
        """Ordinary dot product"""
        if numpy.isscalar(other):
            return self * other
        else:
            return self @ other

    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Subclasses override this to provide format-specific counts.
        Public access is via :attr:`nnz` (no axis) or
        :meth:`spmatrix.getnnz` (matrix-only, axis-aware).
        """
        raise NotImplementedError

    def nonzero(self):
        """Indices of the non-zero elements.

        Returns a tuple of cupy.ndarrays, one per dimension -- ``(row,
        col)`` for a 2-D array/matrix or ``(col,)`` for a 1-D array --
        matching :meth:`scipy.sparse._base._spbase.nonzero`.  Explicit
        zeros are excluded.
        """
        A = self.tocoo()
        if not A.has_canonical_format:
            A.sum_duplicates()
        nz_mask = A.data != 0
        if A.ndim == 1:
            return (A.col[nz_mask],)
        return (A.row[nz_mask], A.col[nz_mask])

    def maximum(self, other):
        return self.tocsr().maximum(other)

    def mean(self, axis=None, dtype=None, out=None):
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the matrix elements. The average is taken
        over all elements in the matrix by default, otherwise over the
        specified axis. `float64` intermediate and return values are used
        for integer inputs.

        Args:
            axis {-2, -1, 0, 1, None}: optional
                Axis along which the mean is computed. The default is to
                compute the mean of all elements in the matrix
                (i.e., `axis` = `None`).
            dtype (dtype): optional
                Type to use in computing the mean. For integer inputs, the
                default is `float64`; for floating point inputs, it is the same
                as the input dtype.
            out (cupy.ndarray): optional
                Alternative output matrix in which to place the result. It must
                have the same shape as the expected output, but the type of the
                output values will be cast if necessary.

        Returns:
            m (cupy.ndarray) : Output array of means

        .. seealso::
            :meth:`scipy.sparse.spmatrix.mean`

        """
        def _is_integral(dtype):
            return (cupy.issubdtype(dtype, cupy.integer) or
                    cupy.issubdtype(dtype, cupy.bool_))

        _sputils.validateaxis(axis)

        res_dtype = self.dtype.type
        integral = _is_integral(self.dtype)

        # output dtype
        if dtype is None:
            if integral:
                res_dtype = cupy.float64
        else:
            res_dtype = cupy.dtype(dtype).type

        # intermediate dtype for summation
        inter_dtype = cupy.float64 if integral else res_dtype
        inter_self = self.astype(inter_dtype)

        if axis is None:
            return (inter_self / cupy.array(
                self.shape[0] * self.shape[1]))\
                .sum(dtype=res_dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            return (inter_self * (1.0 / self.shape[0])).sum(
                axis=0, dtype=res_dtype, out=out)
        else:
            return (inter_self * (1.0 / self.shape[1])).sum(
                axis=1, dtype=res_dtype, out=out)

    def minimum(self, other):
        return self.tocsr().minimum(other)

    def multiply(self, other):
        """Point-wise multiplication by another matrix"""
        if issparse(other):
            other = other.tocsr()
        return self.tocsr().multiply(other)

    def power(self, n, dtype=None):
        return self.tocsr().power(n, dtype=dtype)

    def reshape(self, *shape, order='C'):
        """Gives a new shape to a sparse matrix without changing its data.

        Args:
            shape (tuple):
                The new shape should be compatible with the original shape.
            order: {'C', 'F'} (optional)
                Read the elements using this index order. 'C' means to read and
                write the elements using C-like index order. 'F' means to read
                and write the elements using Fortran-like index order. Default:
                C.

        Returns:
            cupyx.scipy.sparse.coo_matrix: sparse matrix

        """
        shape = _sputils.check_shape(
            shape, self.shape, allow_nd=self._allow_nd)

        if shape == self.shape:
            return self

        return self.tocoo().reshape(shape, order=order)

    def setdiag(self, values, k=0):
        """Set diagonal or off-diagonal elements of the array.

        Args:
            values (cupy.ndarray): New values of the diagonal elements.
                Values may have any length. If the diagonal is longer than
                values, then the remaining diagonal entries will not be set.
                If values is longer than the diagonal, then the remaining
                values are ignored. If a scalar value is given, all of the
                diagonal is set to it.
            k (int, optional): Which diagonal to set, corresponding to elements
                a[i, i+k]. Default: 0 (the main diagonal).
        """
        raise NotImplementedError

    def sum(self, axis=None, dtype=None, out=None):
        """Sums the matrix elements over a given axis.

        Args:
            axis (int or ``None``): Axis along which the sum is computed.
                If it is ``None``, it computes the sum of all the elements.
                Select from ``{None, 0, 1, -2, -1}``.
            dtype: The type of returned matrix. If it is not specified, type
                of the array is used.
            out (cupy.ndarray): Output matrix.

        Returns:
            cupy.ndarray: Summed array.

        .. seealso::
           :meth:`scipy.sparse.spmatrix.sum`

        """
        # This implementation uses multiplication, though it is not efficient
        # for some matrix types. These should override this function.

        if self.ndim == 1:
            # The only axis reduces everything to a scalar.  Summing the
            # stored data (duplicates included) equals the dense sum
            # because implicit zeros contribute nothing.
            _sputils.validate_axis_1d(axis)
            ret = self.data.sum(dtype=dtype)
            if out is not None:
                if out.shape != ret.shape:
                    raise ValueError('dimensions do not match')
                _core.elementwise_copy(ret, out)
                return out
            return ret

        m, n = self.shape

        if self.ndim == 2 and axis == (0, 1):
            axis = None

        _sputils.validateaxis(axis)

        if axis is None:
            return self.dot(cupy.ones(n, dtype=self.dtype)).sum(
                dtype=dtype, out=out)

        if axis < 0:
            axis += 2

        if isinstance(self, sparray):
            # Arrays: reduction along an axis returns 1D
            if axis == 0:
                ret = self.T.dot(cupy.ones(m, dtype=self.dtype))
            else:
                ret = self.dot(cupy.ones(n, dtype=self.dtype))
        else:
            # Matrices: keep 2D shape
            if axis == 0:
                ret = self.T.dot(
                    cupy.ones(m, dtype=self.dtype)).reshape(1, n)
            else:
                ret = self.dot(
                    cupy.ones(n, dtype=self.dtype)).reshape(m, 1)

        if out is not None:
            if out.shape != ret.shape:
                raise ValueError('dimensions do not match')
            _core.elementwise_copy(ret, out)
            return out
        elif dtype is not None:
            return ret.astype(dtype, copy=False)
        else:
            return ret

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


class sparray:
    """Namespace mixin for sparse array classes.

    Sparse array classes follow NumPy semantics: ``*`` is element-wise
    multiplication and ``**`` is element-wise power.  Use ``@`` for
    matrix multiplication.

    .. seealso:: :class:`scipy.sparse.sparray`
    """
    pass


class spmatrix:
    """Mixin for sparse matrix classes.

    Sparse matrix classes follow legacy ``numpy.matrix`` semantics:
    ``*`` is matrix multiplication and ``**`` is matrix power.  Provides
    backward-compatibility methods (``.A``, ``.H``, ``getrow``,
    ``getcol``, etc.) that do not exist on sparse arrays.  These APIs
    are deprecated in favor of the sparse array interface.

    .. seealso:: :class:`scipy.sparse.spmatrix`
    """

    def __init__(self, *args, maxprint=50, **kwargs):
        self.maxprint = maxprint
        # Cooperative MI: forward to the next __init__ in the MRO.
        nxt = super().__init__
        if nxt is not object.__init__:
            nxt(*args, **kwargs)
        elif args or kwargs:
            # Direct instantiation of ``spmatrix(args)`` with no concrete
            # format subclass; scipy raises here, so do the same.
            raise TypeError(
                'cannot instantiate spmatrix directly; use a format '
                'subclass such as csr_matrix')

    # Matrix semantics: * is matmul (overrides _spbase element-wise default)
    def __mul__(self, other):
        return self._matmul_dispatch(other)

    def __rmul__(self, other):
        return self._rmatmul_dispatch(other)

    def __pow__(self, other):
        """Calculates n-th power of the matrix.

        This method calculates n-th power of a given matrix. The matrix must
        be a squared matrix, and a given exponent must be an integer.

        Args:
            other (int): Exponent.

        Returns:
            cupyx.scipy.sparse.spmatrix: A sparse matrix representing n-th
            power of this matrix.

        """
        m, n = self.shape
        if m != n:
            raise TypeError('matrix is not square')
        if not isinstance(other, numbers.Integral):
            raise ValueError("exponent must be an integer")

        if _util.isintlike(other):
            other = int(other)
            if other < 0:
                raise ValueError('exponent must be >= 0')

            if other == 0:
                import cupyx.scipy.sparse
                return cupyx.scipy.sparse.identity(
                    m, dtype=self.dtype, format='csr')
            elif other == 1:
                return self.copy()
            else:
                tmp = self.__pow__(other // 2)
                if other % 2:
                    return self * tmp * tmp
                else:
                    return tmp * tmp
        elif _util.isscalarlike(other):
            raise ValueError('exponent must be an integer')
        else:
            return NotImplemented

    @property
    def _csr_container(self):
        from cupyx.scipy.sparse._csr import csr_matrix
        return csr_matrix

    @property
    def _csc_container(self):
        from cupyx.scipy.sparse._csc import csc_matrix
        return csc_matrix

    @property
    def _coo_container(self):
        from cupyx.scipy.sparse._coo import coo_matrix
        return coo_matrix

    @property
    def _dia_container(self):
        from cupyx.scipy.sparse._dia import dia_matrix
        return dia_matrix

    @property
    def A(self):
        """Dense ndarray representation of this matrix.

        .. deprecated:: 15.0
           Use :meth:`~cupyx.scipy.sparse.spmatrix.toarray` instead.

        """
        warnings.warn(
            "`spmatrix.A` is deprecated; use `.toarray()` instead.",
            DeprecationWarning, stacklevel=2)
        return self.toarray()

    @property
    def H(self):
        """Hermitian (conjugate) transpose of this matrix.

        .. deprecated:: 15.0
           Use ``.T.conj()`` instead.

        """
        warnings.warn(
            "`spmatrix.H` is deprecated; use `.T.conj()` instead.",
            DeprecationWarning, stacklevel=2)
        return self.transpose().conj()

    def get_shape(self):
        """Return the shape of the matrix."""
        return self._shape

    def set_shape(self, shape):
        """Set the shape of the matrix in-place."""
        # Match scipy 1.17: build the reshaped matrix and swap __dict__
        # so the change is visible on the original object.
        new_self = self.reshape(shape).asformat(self.format)
        self.__dict__ = new_self.__dict__

    shape = property(
        fget=get_shape, fset=set_shape, doc='Shape of the matrix.')

    def asfptype(self):
        """Upcasts matrix to a floating point format.

        When the matrix has floating point type, the method returns itself.
        Otherwise it makes a copy with floating point type and the same
        format.

        Returns:
            cupyx.scipy.sparse.spmatrix: A matrix with float type.
        """
        if self.dtype.kind == 'f':
            return self
        typ = numpy.promote_types(self.dtype, 'f')
        return self.astype(typ)

    def getH(self):
        """Hermitian (conjugate) transpose of this matrix."""
        return self.transpose().conj()

    def getrow(self, i):
        """Return a copy of row ``i`` as a (1 x n) sparse row vector.

        Matrix-only API; for sparse arrays use ``A[i]`` (or ``A[[i], :]``
        for a 2-D result).
        """
        return self._getrow(i)

    def getcol(self, j):
        """Return a copy of column ``j`` as a (m x 1) sparse column vector.

        Matrix-only API; for sparse arrays use ``A[:, j]`` (or
        ``A[:, [j]]`` for a 2-D result).
        """
        return self._getcol(j)

    def getformat(self):
        """Return the format string of this matrix (e.g. ``'csr'``)."""
        return self.format

    def getmaxprint(self):
        """Return the maximum number of stored values shown in ``__str__``."""
        return self.maxprint

    def getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Args:
            axis (None, 0, or 1): Select between the number of values
                across the whole matrix, in each column (axis=0), or in
                each row (axis=1).
        """
        return self._getnnz(axis=axis)


def issparse(x):
    """Checks if a given matrix is a sparse matrix or array.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.spmatrix`
        or :class:`cupyx.scipy.sparse.sparray`.

    """
    return isinstance(x, _spbase)


def isspmatrix(x):
    """Checks if a given matrix is a sparse matrix (not array).

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.spmatrix`.

    """
    return isinstance(x, spmatrix)


isdense = _util.isdense
