import numpy

import cupy
from cupy.core.core cimport ndarray


def polysub(a1, a2):
    """Computes the difference of two polynomials.

    Args:
        a1 (scalar, cupy.ndarray or cupy.poly1d): first input polynomial.
        a2 (scalar, cupy.ndarray or cupy.poly1d): second input polynomial.

    Returns:
        cupy.ndarray or cupy.poly1d: The difference of the inputs.

    .. seealso:: :func:`numpy.polysub`

    """
    truepoly = False
    if isinstance(a1, poly1d):
        a1 = a1._coeffs
        truepoly = True
    if isinstance(a2, poly1d):
        a2 = a2._coeffs
        truepoly = True
    a1 = cupy.atleast_1d(a1)
    a2 = cupy.atleast_1d(a2)
    if a1.shape[0] < a2.shape[0]:
        a1 = cupy.pad(a1, (a2.shape[0] - a1.shape[0], 0))
    elif a1.shape[0] > a2.shape[0]:
        a2 = cupy.pad(a2, (a1.shape[0] - a2.shape[0], 0))
    val = a1 - a2
    if truepoly:
        val = poly1d(val)
    return val


cdef class poly1d:
    """A one-dimensional polynomial class.

    Args:
        c_or_r (array_like): The polynomial's
         coefficients in decreasing powers
        r (bool, optional): If True, ``c_or_r`` specifies the
            polynomial's roots; the default is False.
        variable (str, optional): Changes the variable used when
            printing the polynomial from ``x`` to ``variable``

    .. seealso:: :func:`numpy.poly1d`

    """
    __hash__ = None

    cdef:
        readonly ndarray _coeffs
        readonly str _variable
        readonly bint _trimmed

    @property
    def coeffs(self):
        if self._trimmed:
            return self._coeffs
        self._coeffs = cupy.trim_zeros(self._coeffs, trim='f')
        if self._coeffs.size == 0:
            self._coeffs = cupy.array([0.])
        self._trimmed = True
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
        return self.coeffs.size - 1

    # TODO(Dahlia-Chehata): implement using cupy.roots
    @property
    def roots(self):
        raise NotImplementedError

    @property
    def r(self):
        return self.roots

    @property
    def c(self):
        return self.coeffs

    @property
    def coef(self):
        return self.coeffs

    @property
    def coefficients(self):
        return self.coeffs

    @property
    def o(self):
        return self.order

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, (numpy.poly1d, poly1d)):
            self._coeffs = cupy.asarray(c_or_r.coeffs)
            self._variable = c_or_r._variable
            self._trimmed = True
            if variable is not None:
                self._variable = variable
            return
        # TODO(Dahlia-Chehata): if r: c_or_r = poly(c_or_r)
        if r:
            raise NotImplementedError
        c_or_r = cupy.atleast_1d(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError('Polynomial must be 1d only.')
        self._coeffs = c_or_r
        self._trimmed = False
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

    # TODO(Dahlia-Chehata): implement using polyadd
    def __add__(self, other):
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polyadd
    def __radd__(self, other):
        raise NotImplementedError

    # TODO(Dahlia-Chehata): implement using polymul
    def __pow__(self, val, modulo):
        if not cupy.isscalar(val) or int(val) != val or val < 0:
            raise ValueError('Power to non-negative integers only.')
        raise NotImplementedError

    def __sub__(self, other):
        return polysub(self, other)

    def __rsub__(self, other):
        return polysub(other, self)

    # TODO(Dahlia-Chehata): use polydiv for non-scalars
    def __truediv__(self, other):
        if cupy.isscalar(other):
            return poly1d(self.coeffs / other)
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            raise NotImplementedError
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return (self.coeffs == other.coeffs).all().get()

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            raise NotImplementedError
        return not self.__eq__(other)

    def __getitem__(self, val):
        if 0 <= val < self._coeffs.size:
            return self._coeffs[-val-1]
        return 0

    def __setitem__(self, key, val):
        if key < 0:
            raise ValueError('Negative powers are not supported.')
        if key >= self._coeffs.size:
            self._coeffs = cupy.pad(self._coeffs,
                                    (key - (self._coeffs.size - 1), 0))
        self._coeffs[-key-1] = val
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
    cpdef get(self, stream=None):
        """Returns a copy of poly1d object on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.
                The default uses CUDA stream object of the current context.

        Returns:
            numpy.poly1d: Copy of poly1d object on host memory.

        """
        return numpy.poly1d(self.coeffs.get(stream=stream),
                            variable=self.variable)

    cpdef set(self, polyin, stream=None):
        """Copies a poly1d object on the host memory to :class:`cupy.poly1d`.

        Args:
            polyin (numpy.poly1d): The source object on the host memory.
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.
                The default uses CUDA stream object of the current context.

        """
        if not isinstance(polyin, numpy.poly1d):
            raise TypeError('Only numpy.poly1d can be set to cupy.poly1d')
        self._variable = polyin.variable
        self.coeffs.set(polyin.coeffs, stream)
