import cupy
from cupyx.scipy.sparse import base
from .util import validateaxis


_ufuncs = [
    'arcsin', 'arcsinh', 'arctan', 'arctanh', 'ceil', 'deg2rad', 'expm1',
    'floor', 'log1p', 'rad2deg', 'rint', 'sign', 'sin', 'sinh', 'sqrt', 'tan',
    'tanh', 'trunc',
]


class _data_matrix(base.spmatrix):

    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        """Data type of the matrix."""
        return self.data.dtype

    def _with_data(self, data, copy=True):
        raise NotImplementedError

    def __abs__(self):
        """Elementwise abosulte."""
        return self._with_data(abs(self.data))

    def __neg__(self):
        """Elementwise negative."""
        return self._with_data(-self.data)

    def astype(self, t):
        """Casts the array to given data type.

        Args:
            dtype: Type specifier.

        Returns:
            A copy of the array with a given type.

        """
        return self._with_data(self.data.astype(t))

    def conj(self, copy=True):
        if cupy.issubdtype(self.dtype, cupy.complexfloating):
            return self._with_data(self.data.conj(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    conj.__doc__ = base.spmatrix.conj.__doc__

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    copy.__doc__ = base.spmatrix.copy.__doc__

    def count_nonzero(self):
        """Returns number of non-zero entries.

        .. note::
           This method counts the actual number of non-zero entories, which
           does not include explicit zero entries.
           Instead ``nnz`` returns the number of entries including explicit
           zeros.

        Returns:
            Number of non-zero entries.

        """
        return cupy.count_nonzero(self.data)

    def power(self, n, dtype=None):
        """Elementwise power function.

        Args:
            n: Exponent.
            dtype: Type specifier.

        """
        if dtype is None:
            data = self.data.copy()
        else:
            data = self.data.astype(dtype, copy=True)
        data **= n
        return self._with_data(data)


def _find_missing_index(ind, n):
    for k, a in enumerate(ind):
        if k != a:
            return k

    k += 1
    if k < n:
        return k
    else:
        return -1


class _minmax_mixin(object):
    """Mixin for min and max methods.
    These are not implemented for dia_matrix, hence the separate class.

    """

    def _min_or_max_axis(self, axis, min_or_max, sum_duplicates, nonzero):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")

        mat = self.tocsc() if axis == 0 else self.tocsr()
        if sum_duplicates:
            mat.sum_duplicates()

        # Do the reudction
        value = mat._minor_reduce(min_or_max, axis, nonzero)

        return value

    def _min_or_max(self, axis, out, min_or_max, sum_duplicates, non_zero):
        if out is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'out' parameter."))

        validateaxis(axis)

        if axis == 0 or axis == 1:
            return self._min_or_max_axis(axis, min_or_max, sum_duplicates,
                                         non_zero)
        else:
            raise ValueError("axis out of range")

    def _arg_min_or_max_axis(self, axis, op, sum_duplicates):
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized "
                             "dimension.")

        mat = self.tocsc() if axis == 0 else self.tocsr()

        if sum_duplicates:
            mat.sum_duplicates()

        # Do the reudction
        value = mat._arg_minor_reduce(op, axis)

        return value

    def _arg_min_or_max(self, axis, out, op, compare, sum_duplicates):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to "
                                 "an empty matrix.")

            if self.nnz == 0:
                return 0
            else:
                zero = self.dtype.type(0)
                mat = self.tocoo()

                if sum_duplicates:
                    mat.sum_duplicates()

                am = op(mat.data)
                m = mat.data[am]

                if compare(m, zero):
                    return mat.row[am] * mat.shape[1] + mat.col[am]
                else:
                    size = cupy.prod(mat.shape)
                    if size == mat.nnz:
                        return am
                    else:
                        ind = mat.row * mat.shape[1] + mat.col
                        zero_ind = _find_missing_index(ind, size)
                        if m == zero:
                            return min(zero_ind, am)
                        else:
                            return zero_ind

        return self._arg_min_or_max_axis(axis, op, sum_duplicates)

    def max(self, axis=None, out=None, sum_duplicates=False, nonzero=False):
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
            sum_duplicates (bool): Flag to indicate that duplicate elements
                should be combined prior to the operation
            nonzero (bool): Return the maximum nonzero value and ignore all
                zero entries. If the dimension has no nonzero values, a zero is
                then returned to indicate that it is the only available value.

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

        return self._min_or_max(axis, out, cupy.max, sum_duplicates, nonzero)

    def min(self, axis=None, out=None, sum_duplicates=False, nonzero=False):
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
            sum_duplicates (bool): Flag to indicate that duplicate elements
                should be combined prior to the operation
            nonzero (bool): Return the minimum nonzero value and ignore all
                zero entries. If the dimension has no nonzero values, a zero is
                then returned to indicate that it is the only available value.

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

        return self._min_or_max(axis, out, cupy.min, sum_duplicates, nonzero)

    def argmax(self, axis=None, out=None, sum_duplicates=False):
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
            sum_duplicates (bool): Flag to indicate that duplicate elements
                should be combined prior to the operation

        Returns:
            (cupy.narray or int): Indices of maximum elements. If array,
                its size along ``axis`` is 1.

        """

        return self._arg_min_or_max(axis, out, cupy.argmax, cupy.greater,
                                    sum_duplicates)

    def argmin(self, axis=None, out=None, sum_duplicates=False):
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
            sum_duplicates (bool): Flag to indicate that duplicate elements
                should be combined prior to the operation

        Returns:
            (cupy.narray or int): Indices of minimum elements. If matrix,
                its size along ``axis`` is 1.

        """

        return self._arg_min_or_max(axis, out, cupy.argmin, cupy.less,
                                    sum_duplicates)


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
