import cupy
from cupy.core import internal
from cupy import _util
from cupyx.scipy.sparse import base
from cupyx.scipy.sparse import coo
from cupyx.scipy.sparse import sputils


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

    def mean(self, axis=None, dtype=None, out=None):
        """Compute the arithmetic mean along the specified axis.

        Args:
            axis (int or ``None``): Axis along which the sum is computed.
                If it is ``None``, it computes the average of all the elements.
                Select from ``{None, 0, 1, -2, -1}``.

        Returns:
            cupy.ndarray: Summed array.

        .. seealso::
           :meth:`scipy.sparse.spmatrix.mean`

        """
        sputils.validateaxis(axis)
        nRow, nCol = self.shape
        data = self.data.copy()

        if axis is None:
            n = nRow * nCol
        elif axis in (0, -2):
            n = nRow
        else:
            n = nCol

        return self._with_data(data / n).sum(axis, dtype, out)

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

    def _min_or_max_axis(self, axis, min_or_max, explicit):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        # Do the reduction
        value = mat._minor_reduce(min_or_max, axis, explicit)
        major_index = cupy.arange(M)

        mask = value != 0
        major_index = cupy.compress(mask, major_index)
        value = cupy.compress(mask, value)

        if axis == 0:
            return coo.coo_matrix(
                (value, (cupy.zeros(len(value)), major_index)),
                dtype=self.dtype, shape=(1, M))
        else:
            return coo.coo_matrix(
                (value, (major_index, cupy.zeros(len(value)))),
                dtype=self.dtype, shape=(M, 1))

    def _min_or_max(self, axis, out, min_or_max, explicit):
        if out is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'out' parameter."))

        sputils.validateaxis(axis)

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
                    assert False
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

        if axis == 0:
            return value[None, :]
        else:
            return value[:, None]

    def _arg_min_or_max(self, axis, out, op, compare):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        sputils.validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to "
                                 "an empty matrix.")

            if self.nnz == 0:
                return 0
            else:
                zero = self.dtype.type(0)
                mat = self.tocoo()

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
