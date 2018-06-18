from cupy import core
from cupy.math import ufunc


i0 = ufunc.create_math_ufunc(
    'cyl_bessel_i0', 1, 'cupy_i0',
    '''Modified Bessel function of the first kind, order 0.

    .. seealso:: :func:`numpy.i0`

    ''')


sinc = core.create_ufunc(
    'cupy_sinc', ('e->e', 'f->f', 'd->d'),
    'out0 = abs(in0) > 1e-9 ? sinpi(in0) / (M_PI * in0) : 1',
    doc='''Elementwise sinc function.

    .. seealso:: :func:`numpy.sinc`

    ''')
