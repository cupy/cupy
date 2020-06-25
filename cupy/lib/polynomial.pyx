import re
import warnings

import ctypes
import numpy

import cupy
from cupy.cuda import device
from cupy.core import syncdetect
from cupy.cuda cimport stream as stream_module


_poly_mat = re.compile(r'[*][*]([0-9]*)')


def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' ' * (len(power) - 1)
        toadd1 = ' ' * (len(partstr) - 1) + power
        if ((len(line2) + len(toadd2) > wrap) or
                (len(line1) + len(toadd1) > wrap)):
            output += line1 + '\n' + line2 + '\n '
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' ' * (len(power) - 1)
            line1 += ' ' * (len(partstr) - 1) + power
    output += line1 + '\n' + line2
    return output + astr[n:]


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

    @property
    def _coeffs(self):
        return self.__dict__['coeffs']

    @_coeffs.setter
    def _coeffs(self, coeffs):
        self.__dict__['coeffs'] = coeffs

    r = roots
    c = coef = coefficients = coeffs
    o = order

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._variable = c_or_r._variable
            self._coeffs = c_or_r._coeffs

            if set(c_or_r.__dict__) - set(self.__dict__):
                msg = ('In the future extra properties will not be copied '
                       'across when constructing one poly1d from another')
                warnings.warn(msg, FutureWarning, stacklevel=2)
                self.__dict__.update(c_or_r.__dict__)

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
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        thestr = '0'
        var = self.variable

        coeffs = cupy.trim_zeros(self.coeffs, trim='f')

        def fmt_float(q):
            s = '%.4g' % q
            if s.endswith('.0000'):
                s = s[:-5]
            return s

        for i, coeff in enumerate(coeffs):
            if not cupy.iscomplex(coeff):
                coefstr = fmt_float(cupy.real(coeff))
            elif cupy.real(coeff) == 0:
                coefstr = '%sj' % fmt_float(cupy.imag(coeff))
            else:
                coefstr = '(%s + %sj)' % (fmt_float(cupy.real(coeff)),
                                          fmt_float(cupy.imag(coeff)))

            power = coeffs.size - 1 - i
            if power == 0:
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    if i == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = '%s %s' % (coefstr, var)
            else:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = '%s**%d' % (var, power,)
                else:
                    newstr = '%s %s**%d' % (coefstr, var, power)

            if i > 0:
                if newstr != '':
                    if newstr.startswith('-'):
                        thestr = "%s - %s" % (thestr, newstr[1:])
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            else:
                thestr = newstr
        return _raise_power(thestr)

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
                raise TypeError('Only numpy.poly1d can be obtained from'
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
            coeffs_cpu = out.coeffs
        else:
            if self.coeffs.size == 0:
                c = numpy.array(self.coeffs.shape, self.coeffs.dtype)
                return numpy.poly1d(c, variable=self.variable)
            coeffs_gpu = self.coeffs
            coeffs_cpu = numpy.empty(self.coeffs.shape, self.coeffs.dtype)
        syncdetect._declare_synchronize()
        ptr = ctypes.c_void_p(coeffs_cpu.__array_interface__['data'][0])
        with self.coeffs.device:
            if stream is not None:
                coeffs_gpu.data.copy_to_host_async(ptr,
                                                   coeffs_gpu.nbytes, stream)
            else:
                stream_ptr = stream_module.get_current_stream_ptr()
                if stream_ptr == 0:
                    coeffs_gpu.data.copy_to_host(ptr, coeffs_gpu.nbytes)
                else:
                    coeffs_gpu.data.copy_to_host_async(ptr, coeffs_gpu.nbytes)
        return numpy.poly1d(coeffs_cpu, variable=self.variable)


warnings.simplefilter('always', numpy.RankWarning)
