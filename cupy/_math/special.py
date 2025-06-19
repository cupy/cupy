from cupy import _core
from cupy._math import ufunc


i0 = ufunc.create_math_ufunc(
    'cyl_bessel_i0', 1, 'cupy_i0',
    '''Modified Bessel function of the first kind, order 0.

    .. seealso:: :func:`numpy.i0`

    ''')


sinc = _core.create_ufunc(
    'cupy_sinc',
    ('e->e', 'f->f', 'd->d',
     ('F->F', 'in0_type pi_in0 = (in0_type) M_PI * in0;'
              'out0 = abs(in0) > 1e-9 ? sin(pi_in0) / (pi_in0) : 1'),
     ('D->D', 'in0_type pi_in0 = (in0_type) M_PI * in0;'
              'out0 = abs(in0) > 1e-9 ? sin(pi_in0) / (pi_in0) : 1')),
    'out0 = abs(in0) > 1e-9 ? sinpi(in0) / (M_PI * in0) : 1',
    doc='''Elementwise sinc function.

    .. seealso:: :func:`numpy.sinc`

    ''')
