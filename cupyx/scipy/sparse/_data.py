from __future__ import annotations

import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util as _sparse_util


_ufuncs = [
    'arcsin', 'arcsinh', 'arctan', 'arctanh', 'ceil', 'deg2rad', 'expm1',
    'floor', 'log1p', 'rad2deg', 'rint', 'sign', 'sin', 'sinh', 'sqrt', 'tan',
    'tanh', 'trunc',
]


class _data_matrix(_base._spbase):

    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        """Data type of the matrix."""
        return self.data.dtype

    @property
    def device(self):
        """CUDA device on which this array resides."""
        return self.data.device

    def _with_data(self, data, copy=True):
        raise NotImplementedError

    def _bool_reduction_carrier(self):
        """Non-mutating float64 carrier for reducing a bool array.

        cuSPARSE has no bool arithmetic, so bool ``sum``/``mean`` reduce
        through a float64 carrier.  Bool duplicates coalesce by logical-or
        (densify semantics, matching :meth:`toarray`), so the carrier is
        deduplicated -- but on a *copy*, never mutating ``self`` (a
        reduction must not compact its operand).  An already-canonical
        array, or a format that cannot hold duplicates (DIA), skips the
        copy; the raw flag is read without a device sync.
        """
        src = self
        if (getattr(self, '_has_canonical_format', None) is not True
                and hasattr(self, 'sum_duplicates')):
            src = self.copy()
            src.sum_duplicates()
        return src._with_data(src.data.astype(cupy.float64), copy=False)

    def __abs__(self):
        """Elementwise absolute."""
        return self._with_data(abs(self.data))

    def __round__(self, ndigits=0):
        """Elementwise rounding (matches :func:`numpy.around`)."""
        return self._with_data(cupy.around(self.data, decimals=ndigits))

    def __neg__(self):
        """Elementwise negative."""
        if self.dtype.kind == 'b':
            # Match scipy 1.17: raise NotImplementedError instead of letting
            # the underlying cupy error surface.
            raise NotImplementedError(
                'negating a boolean sparse array is not supported')
        return self._with_data(-self.data)

    @staticmethod
    def _scalar_op_dtype(self_dtype, other):
        """Pick the dtype for ``self.data * other`` / ``... / other``.

        numpy's natural promotion can land outside the cuSPARSE-supported
        set (e.g. ``bool * int -> int64``).  Upcast to ``float64`` in
        that case so the result remains usable.  Mirrors scipy which
        promotes bool / int sparse to float on division.
        """
        out = np.result_type(self_dtype, other)
        if not _sputils.is_sparse_data_dtype(out):
            out = np.dtype(np.float64)
        return out

    def __imul__(self, other):
        # In-place scalar multiply mutates ``self.data`` to preserve
        # object identity.  Non-scalar falls through to NotImplemented
        # so Python rebinds via ``self = self * other``.  Unlike scipy,
        # which raises ``UFuncTypeError`` on out-of-set promotions
        # (e.g. ``bool *= int``), CuPy reassigns ``self.data`` to a
        # cuSPARSE-supported dtype.  Identity of ``self`` is preserved;
        # identity of ``self.data`` is not.
        if _sparse_util.isscalarlike(other):
            new_dtype = self._scalar_op_dtype(self.dtype, other)
            if new_dtype != self.dtype:
                self.data = self.data.astype(new_dtype) * other
            else:
                self.data *= other
            return self
        return NotImplemented

    def __itruediv__(self, other):
        # In-place scalar division mutates ``self.data``.  See
        # ``__imul__`` for the upcast rationale; division additionally
        # promotes int dividends to float (``int / 2`` is ``float``
        # in Python and ``self.data /= 2`` would otherwise raise on
        # int data).
        if _sparse_util.isscalarlike(other):
            recip = 1.0 / other
            new_dtype = self._scalar_op_dtype(self.dtype, recip)
            if new_dtype != self.dtype:
                self.data = self.data.astype(new_dtype) * recip
            else:
                self.data *= recip
            return self
        return NotImplemented

    def astype(self, dtype, copy=True):
        """Cast the array elements to a specified type.

        Args:
            dtype: Target dtype.
            copy (bool): If ``True`` (default), the returned array does
                not share memory with ``self``.  If ``False``, ``self``
                is returned unchanged when the dtype already matches.

        Returns:
            Sparse object with the requested dtype and the same format.
        """
        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            return self._with_data(self.data.astype(dtype, copy=copy))
        if copy:
            return self.copy()
        return self

    def conj(self, copy=True):
        if cupy.issubdtype(self.dtype, cupy.complexfloating):
            return self._with_data(self.data.conj(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    conj.__doc__ = _base._spbase.conj.__doc__

    @property
    def real(self):
        return self._with_data(self.data.real)

    @property
    def imag(self):
        return self._with_data(self.data.imag)

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    copy.__doc__ = _base._spbase.copy.__doc__

    def count_nonzero(self, axis=None):
        """Number of non-zero entries.

        Unlike :attr:`nnz` (length of ``data``), this counts only true
        non-zero values; explicit-zero stored entries are excluded.

        Args:
            axis (``None``, optional): Only ``None`` is handled here;
                CSR/CSC/COO override this to support ``0``, ``1``,
                ``-1``, ``-2``.

        Returns:
            int or cupy.ndarray: Scalar count when ``axis`` is
            ``None``; otherwise a 1-D array (from format override).
        """
        # Match scipy: dedup in place before counting.
        if hasattr(self, 'sum_duplicates'):
            self.sum_duplicates()
        if axis is None:
            return int(cupy.count_nonzero(self.data))
        raise NotImplementedError(
            'axis-aware count_nonzero is not implemented for '
            f'{type(self).__name__}')

    def mean(self, axis=None, dtype=None, out=None):
        """Compute the arithmetic mean along the specified axis.

        Args:
            axis (int or ``None``): Axis along which the sum is computed.
                If it is ``None``, it computes the average of all the elements.
                Select from ``{None, 0, 1, -2, -1}``.

        Returns:
            cupy.ndarray: The computed mean.

        .. seealso::
           :meth:`scipy.sparse.spmatrix.mean`

        """
        # Scale by the Python-level reciprocal like scipy does: the mean
        # of a zero-length axis then raises ZeroDivisionError instead of
        # silently returning 0.
        if self.dtype == bool:
            # Densify-consistent mean: reduce the coalesced float64 carrier
            # so bool duplicates OR together (matching sum()/toarray()).
            # (scipy's mean instead counts duplicates -- inconsistent with
            # its own coalescing sum -- so this deliberately diverges from
            # scipy for a non-canonical bool array.)
            return self._bool_reduction_carrier().mean(
                axis=axis, dtype=dtype, out=out)
        if self.ndim == 1:
            # Collapse the single axis to a scalar mean directly, without
            # building a throwaway sparse object.  Accumulate the sum in
            # floating point and scale *before* the final cast: casting each
            # ``1/N``-scaled term to an integer ``dtype`` mid-reduction would
            # truncate it (matches the 2-D path and numpy).  ``out=`` is
            # forwarded to that final reduction, which validates its shape;
            # the Python-level ``1.0 / N`` raises ZeroDivisionError on an
            # empty axis, like scipy.
            _sputils.validate_axis_1d(axis)
            total = self.data.sum() * (1.0 / self.shape[0])
            return total.sum(dtype=dtype, out=out)

        axis = _sputils.collapse_2d_axis(axis)
        _sputils.validateaxis(axis)
        nRow, nCol = self.shape
        if axis is None:
            n = nRow * nCol
        elif axis in (0, -2):
            n = nRow
        else:
            n = nCol

        return self._with_data(self.data * (1.0 / n)).sum(axis, dtype, out)

    def power(self, n, dtype=None):
        """Elementwise power function.

        Args:
            n: Exponent.
            dtype: Type specifier.

        """
        # Non-scalar check must come first: ``n == 0`` on an array
        # produces a bool array and ``if`` on that raises.
        if not _sparse_util.isscalarlike(n):
            raise NotImplementedError('input is not scalar')
        if n == 0:
            raise NotImplementedError(
                'zero power is not supported as it would densify the '
                'matrix.\n'
                'Use cupy.ones(A.shape, dtype=A.dtype) for this case.')
        if dtype is None:
            data = self.data.copy()
        else:
            data = self.data.astype(dtype, copy=True)
        data **= n
        return self._with_data(data)


def _find_missing_index(ind, n):
    positions = cupy.arange(ind.size)
    diff = ind != positions
    return cupy.where(
        diff.any(),
        diff.argmax(),
        cupy.asarray(ind.size if ind.size < n else -1))


def _non_zero_cmp(mat, am, zero, m):
    size = np.prod(mat.shape)
    if size == mat.nnz:
        return am
    else:
        ind = mat.row * mat.shape[1] + mat.col
        zero_ind = _find_missing_index(ind, size)
        return cupy.where(
            m == zero,
            cupy.minimum(zero_ind, am),
            zero_ind)


class _minmax_mixin:
    """Mixin for min and max methods.
    These are not implemented for dia_matrix, hence the separate class.

    """

    def _min_or_max_axis(self, axis, min_or_max, explicit):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        # Do the reduction
        value = mat._minor_reduce(min_or_max, axis, explicit)
        idx_dtype = _sputils.get_index_dtype(maxval=M)
        major_index = cupy.arange(M, dtype=idx_dtype)

        mask = value != 0
        major_index = cupy.compress(mask, major_index)
        value = cupy.compress(mask, value)

        n = len(value)
        zeros = cupy.zeros(n, dtype=idx_dtype)
        value = value.astype(self.dtype, copy=False)
        # Use the appropriate container so the result inherits the array
        # vs matrix type from ``self``.
        coo_cls = self._coo_container
        if isinstance(self, _base.sparray):
            # Sparse arrays reduce a dimension: return a 1-D coo_array
            # of shape ``(M,)`` (matching scipy), stored as a (1, M) row
            # vector (row=zeros, col=major_index).
            return coo_cls._from_parts(value, zeros, major_index, shape=(M,))
        # Matrices keep the legacy 2-D shape (1, M) / (M, 1).
        row, col = (zeros, major_index) if axis == 0 else (major_index, zeros)
        shape = (1, M) if axis == 0 else (M, 1)
        return coo_cls._from_parts(value, row, col, shape=shape)

    def _min_or_max(self, axis, out, min_or_max, explicit):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        if self.ndim == 1:
            # The only axis reduces everything to a scalar.
            _sputils.validate_axis_1d(axis)
            axis = None
        else:
            _sputils.validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")

            zero = cupy.zeros((), dtype=self.dtype)
            if self.nnz == 0:
                return zero
            self.sum_duplicates()
            m = min_or_max(self.data)
            if explicit:
                return m
            if self.nnz != internal.prod(self.shape):
                if min_or_max is cupy.min:
                    m = cupy.minimum(zero, m)
                elif min_or_max is cupy.max:
                    m = cupy.maximum(zero, m)
                else:
                    raise AssertionError(
                        f'unexpected min_or_max ufunc: {min_or_max}')
            return m

        if axis < 0:
            axis += 2

        return self._min_or_max_axis(axis, min_or_max, explicit)

    def _arg_min_or_max_axis(self, axis, op):
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized "
                             "dimension.")

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        # Do the reduction
        value = mat._arg_minor_reduce(op, axis)

        # Sparse arrays return a 1-D ndarray; sparse matrices keep
        # the legacy 2-D shape (matching scipy).
        if isinstance(self, _base.sparray):
            return value
        if axis == 0:
            return value[None, :]
        else:
            return value[:, None]

    def _arg_min_or_max(self, axis, out, op, compare):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        if self.ndim == 1:
            # The only axis reduces everything to a scalar index.
            _sputils.validate_axis_1d(axis)
            axis = None
        else:
            _sputils.validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to "
                                 "an empty matrix.")

            if self.nnz == 0:
                return 0
            else:
                zero = cupy.asarray(self.dtype.type(0))
                mat = self.tocoo()

                mat.sum_duplicates()

                am = op(mat.data)
                m = mat.data[am]

                if mat.ndim == 1:
                    # Use the (1, N) backing: ``row`` is all-zeros, so the
                    # flat index reduces to ``col`` and the 2-D helper
                    # applies unchanged.
                    mat = mat._as_2d()
                return cupy.where(
                    compare(m, zero), mat.row[am] * mat.shape[1] + mat.col[am],
                    _non_zero_cmp(mat, am, zero, m))

        if axis < 0:
            axis += 2

        return self._arg_min_or_max_axis(axis, op)

    def max(self, axis=None, out=None, *, explicit=False):
        """Returns the maximum of the matrix or maximum along an axis.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the sum is computed. The default is to
                compute the maximum over all the matrix elements, returning
                a scalar (i.e. ``axis`` = ``None``).
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except
                for the default value, as this argument is not used.
            explicit (bool): Return the maximum value explicitly specified and
                ignore all implicit zero entries. If the dimension has no
                explicit values, a zero is then returned to indicate that it is
                the only implicit value. This parameter is experimental and may
                change in the future.

        Returns:
            (cupy.ndarray or float): Maximum of ``a``. If ``axis`` is
            ``None``, the result is a scalar value. If ``axis`` is given,
            the result is an array of dimension ``a.ndim - 1``. This
            differs from numpy for computational efficiency.

        .. seealso:: min : The minimum value of a sparse matrix along a given
          axis.
        .. seealso:: numpy.matrix.max : NumPy's implementation of ``max`` for
          matrices

        """
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.max'.format(
                self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.max, explicit)

    def min(self, axis=None, out=None, *, explicit=False):
        """Returns the minimum of the matrix or maximum along an axis.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the sum is computed. The default is to
                compute the minimum over all the matrix elements, returning
                a scalar (i.e. ``axis`` = ``None``).
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.
            explicit (bool): Return the minimum value explicitly specified and
                ignore all implicit zero entries. If the dimension has no
                explicit values, a zero is then returned to indicate that it is
                the only implicit value. This parameter is experimental and may
                change in the future.

        Returns:
            (cupy.ndarray or float): Minimum of ``a``. If ``axis`` is
            None, the result is a scalar value. If ``axis`` is given, the
            result is an array of dimension ``a.ndim - 1``. This differs
            from numpy for computational efficiency.

        .. seealso:: max : The maximum value of a sparse matrix along a given
          axis.
        .. seealso:: numpy.matrix.min : NumPy's implementation of 'min' for
          matrices

        """
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.min'.format(
                self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.min, explicit)

    def argmax(self, axis=None, out=None):
        """Returns indices of maximum elements along an axis.

        Implicit zero elements are taken into account. If there are several
        maximum values, the index of the first occurrence is returned. If
        ``NaN`` values occur in the matrix, the output defaults to a zero entry
        for the row/column in which the NaN occurs.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the argmax is computed. If ``None`` (default),
                index of the maximum element in the flatten data is returned.
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.

        Returns:
            (cupy.narray or int): Indices of maximum elements. If array,
            its size along ``axis`` is 1.

        """
        return self._arg_min_or_max(axis, out, cupy.argmax, cupy.greater)

    def argmin(self, axis=None, out=None):
        """
        Returns indices of minimum elements along an axis.

        Implicit zero elements are taken into account. If there are several
        minimum values, the index of the first occurrence is returned. If
        ``NaN`` values occur in the matrix, the output defaults to a zero entry
        for the row/column in which the NaN occurs.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the argmin is computed. If ``None`` (default),
                index of the minimum element in the flatten data is returned.
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.

        Returns:
            (cupy.narray or int): Indices of minimum elements. If matrix,
            its size along ``axis`` is 1.

        """
        return self._arg_min_or_max(axis, out, cupy.argmin, cupy.less)


def _install_ufunc(func_name):

    def f(self):
        ufunc = getattr(cupy, func_name)
        result = ufunc(self.data)
        return self._with_data(result)

    f.__doc__ = 'Elementwise %s.' % func_name
    f.__name__ = func_name

    setattr(_data_matrix, func_name, f)


def _install_ufuncs():
    for func_name in _ufuncs:
        _install_ufunc(func_name)


_install_ufuncs()
