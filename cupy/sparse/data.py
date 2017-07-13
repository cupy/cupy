import cupy
from cupy.sparse import base


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

    def _with_data(self, data):
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
