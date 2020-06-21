import re
import warnings

import cupy
from cupy.core.overrides import set_module


@set_module('cupy')
class RankWarning(UserWarning):
    """Issued when the Vandermonde matrix is rank deficient.

    .. seealso:: :func:`numpy.RankWarning`

    """
    pass


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


@set_module('cupy')
class poly1d(object):
    """A one-dimensional polynomial class.

    Args:
        c_or_r (cupy.ndarray or cupy.poly1d): The polynomial's
         coefficients in decreasing powers
        r (bool, optional): If True, `c_or_r` specifies the
            polynomial's roots; the default is False.
        variable (str, optional): Changes the variable used when
            printing the polynomial from `x` to `variable`

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

    # TODO
    @property
    def roots(self):
        pass

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
        # TODO: if r: c_or_r = poly(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError('Polynomial must be 1d only.')
        c_or_r = cupy.trim_zeros(c_or_r, trim='f')
        if c_or_r.size == 0:
            c_or_r = cupy.array([0.])
        self._coeffs = c_or_r
        if variable is None:
            variable = 'x'
        self._variable = variable

    def __array__(self, t=None):
        if t:
            return cupy.asarray(self.coeffs, t)
        return self.coeffs

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

    # TODO
    def __call__(self, val):
        pass

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    # TODO
    def __mul__(self, other):
        if cupy.isscalar(other):
            return poly1d(self.coeffs * other)

    # TODO
    def __rmul__(self, other):
        if cupy.isscalar(other):
            return poly1d(other * self.coeffs)

    # TODO
    def __add__(self, other):
        pass

    # TODO
    def __radd__(self, other):
        pass

    # TODO
    def __pow__(self, val):
        if not cupy.isscalar(val) or int(val) != val or val < 0:
            raise ValueError('Power to non-negative integers only.')

    # TODO
    def __sub__(self, other):
        pass

    # TODO
    def __rsub__(self, other):
        pass

    # TODO
    def __div__(self, other):
        if cupy.isscalar(other):
            return poly1d(self.coeffs / other)

    __truediv__ = __div__

    # TODO
    def __rdiv__(self, other):
        if cupy.isscalar(other):
            return poly1d(other / self.coeffs)

    __rtruediv__ = __rdiv__

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return (self.coeffs == other.coeffs).all()

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
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
        pass

    def deriv(self, m=1):
        pass


warnings.simplefilter('always', RankWarning)
