import numpy

import cupy


class poly1d(object):
    """A one-dimensional polynomial class.

    Args:
        c_or_r (array_like): The polynomial's
         coefficients in decreasing powers
        r (bool, optional): If True, ```c_or_r`` specifies the
            polynomial's roots; the default is False.
        variable (str, optional): Changes the variable used when
            printing the polynomial from ``x`` to ``variable``

    .. seealso:: :func:`numpy.poly1d`

    """
    __hash__ = None

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is not self._coeffs:
            raise AttributeError('Cannot set attribute')

    @property
    def variable(self):
        return self._variable

    @property
    def order(self):
        return self._coeffs.size - 1

    # TODO(Dahlia-Chehata): implement using cupy.roots
    @property
    def roots(self):
        raise NotImplementedError

    r = roots
    c = coef = coefficients = coeffs
    o = order

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._variable = c_or_r._variable
            self._coeffs = c_or_r._coeffs

            if variable is not None:
                self._variable = variable
            return
        # TODO(Dahlia-Chehata): if r: c_or_r = poly(c_or_r)
        if c_or_r.ndim < 1:
            c_or_r = cupy.atleast_1d(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError('Polynomial must be 1d only.')
        c_or_r = cupy.trim_zeros(c_or_r, trim='f')
        if c_or_r.size == 0:
            c_or_r = cupy.array([0.])
        self._coeffs = c_or_r
        if variable is None:
            variable = 'x'
        self._variable = variable

    def __array__(self, dtype=None):
        raise TypeError(
            'Implicit conversion to a NumPy array is not allowed. '
            'Please use `.get()` to construct a NumPy array explicitly.')

    def __repr__(self):
        return repr(self.get())

    def __len__(self):
        return self.order

    def __str__(self):
        return str(self.get())

    # TODO(Dahlia-Chehata): implement using polyval
    def __call__(self, val):
        raise NotImplementedError

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    # TODO(Dahlia-Chehata): use polymul for non-scalars
    def __mul__(self, other):
        if cupy.isscalar(other):
            return poly1d(self.coeffs * other)
        raise NotImplementedError

    # TODO(Dahlia-Chehata): use polymul for non-scalars
    def __rmul__(self, other):
        if cupy.isscalar(other):
            return poly1d(other * self.coeffs)
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polyadd
    def __add__(self, other):
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polyadd
    def __radd__(self, other):
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polymul
    def __pow__(self, val):
        if not cupy.isscalar(val) or int(val) != val or val < 0:
            raise ValueError('Power to non-negative integers only.')
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polysub
    def __sub__(self, other):
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polysub
    def __rsub__(self, other):
        raise NotImplementedError

    # TODO(Dahlia-Chehata): use polydiv for non-scalars
    def __div__(self, other):
        if cupy.isscalar(other):
            return poly1d(self.coeffs / other)
        raise NotImplementedError

    __truediv__ = __div__

    # TODO(Dahlia-Chehata): use polydiv for non-scalars
    def __rdiv__(self, other):
        if cupy.isscalar(other):
            return poly1d(other / self.coeffs)
        raise NotImplementedError

    __rtruediv__ = __rdiv__

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            raise NotImplementedError
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return (self.coeffs == other.coeffs).all()

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            raise NotImplementedError
        return not self.__eq__(other)

    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order or val < 0:
            return 0
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError('Negative powers are not supported.')
        if key > self.order:
            zeroz = cupy.zeros(key - self.order, self.coeffs.dtype)
            self._coeffs = cupy.concatenate((zeroz, self.coeffs))
            ind = 0
        self._coeffs[ind] = val
        return

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        raise NotImplementedError

    def deriv(self, m=1):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Cupy specific attributes and methods
    # -------------------------------------------------------------------------
    def get(self, stream=None, out=None):
        """Returns a copy of poly1d object on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.
                The default uses CUDA stream object of the current context.
            out (numpy.poly1d): Output object. In order to enable asynchronous
                copy, the underlying memory should be a pinned memory.

        Returns:
            numpy.poly1d: Copy of poly1d object on host memory.

        """
        if out is not None:
            if not isinstance(out, numpy.poly1d):
                raise TypeError('Only numpy.poly1d can be obtained from '
                                'cupy.poly1d')
            if self.coeffs.dtype != out.coeffs.dtype:
                raise TypeError(
                    '{} poly1d cannot be obtained from {} poly1d'.format(
                        out.coeffs.dtype, self.coeffs.dtype))
            if self.coeffs.shape != out.coeffs.shape:
                raise ValueError(
                    'Shape mismatch. Expected shape: {}, '
                    'actual shape: {}'.format(self.coeffs.shape,
                                              out.coeffs.shape))
            out.coeffs = self.coeffs.get()
            return out
        return numpy.poly1d(self.coeffs.get(stream=stream),
                            variable=self.variable)
