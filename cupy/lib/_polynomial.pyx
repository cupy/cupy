import numbers

import numpy

from cupy._core.core cimport ndarray

import cupy
from cupy.lib import _routines_poly


cdef class poly1d:
    """A one-dimensional polynomial class.

    .. note::
        This is a counterpart of an old polynomial class in NumPy.
        Note that the new NumPy polynomial API
        (:mod:`numpy.polynomial.polynomial`)
        has different convention, e.g. order of coefficients is reversed.

    Args:
        c_or_r (array_like): The polynomial's
         coefficients in decreasing powers
        r (bool, optional): If True, ``c_or_r`` specifies the
            polynomial's roots; the default is False.
        variable (str, optional): Changes the variable used when
            printing the polynomial from ``x`` to ``variable``

    .. seealso:: :class:`numpy.poly1d`

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
            self._coeffs = cupy.array([0.], self._coeffs.dtype)
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

    @property
    def roots(self):
        return _routines_poly.roots(self._coeffs)

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

    @property
    def __cuda_array_interface__(self):
        return self.coeffs.__cuda_array_interface__

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

    def __call__(self, val):
        return _routines_poly.polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    # TODO(kataoka): Rename `self`. It is misleading for cdef class.
    def __mul__(self, other):
        if cupy.isscalar(other):
            # case: poly1d * python scalar
            # the return type of cupy.polymul output is
            # inconsistent with NumPy's output for this case.
            return poly1d(self.coeffs * other)
        if isinstance(self, numpy.generic):
            # case: numpy scalar * poly1d
            # poly1d addition and subtraction don't support this case
            # so it is not supported here for consistency purposes
            # between polyarithmetic routines
            raise TypeError('Numpy scalar and poly1d multiplication'
                            ' is not supported currently.')
        if cupy.isscalar(self) and isinstance(other, poly1d):
            # case: python scalar * poly1d
            # the return type of cupy.polymul output is
            # inconsistent with NumPy's output for this case
            # So casting python scalar is required.
            self = other._coeffs.dtype.type(self)
        return _routines_poly.polymul(self, other)

    def __add__(self, other):
        if isinstance(self, numpy.generic):
            # for the case: numpy scalar + poly1d
            raise TypeError('Numpy scalar and poly1d '
                            'addition is not supported')
        return _routines_poly.polyadd(self, other)

    def __pow__(self, val, modulo):
        if not cupy.isscalar(val) or int(val) != val or val < 0:
            raise ValueError('Power to non-negative integers only.')
        if not isinstance(val, numbers.Integral):
            raise TypeError('float object cannot be interpreted as an integer')

        base = self.coeffs
        dtype = base.dtype

        if dtype.kind == 'c':
            base = base.astype(numpy.complex128, copy=False)
        elif dtype.kind == 'f' or dtype == numpy.uint64:
            base = base.astype(numpy.float64, copy=False)
        else:
            # use Python int here for cross-platform portability
            int_dtype = numpy.promote_types(dtype, int)
            base = base.astype(int_dtype, copy=False)
        return poly1d(_routines_poly._polypow(base, val))

    def __sub__(self, other):
        if isinstance(self, numpy.generic):
            # for the case: numpy scalar - poly1d
            raise TypeError('Numpy scalar and poly1d '
                            'subtraction is not supported')
        return _routines_poly.polysub(self, other)

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
        return (self.coeffs == other.coeffs).all().get()[()]

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
