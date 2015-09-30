import six

from cupy import elementwise
from cupy.math import ufunc


add = ufunc.create_arithmetic(
    'add', '+', '|',
    '''Adds two arrays elementwise.

    .. seealso:: :data:`numpy.add`

    ''')


reciprocal = elementwise.create_ufunc(
    'cupy_reciprocal',
    ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q',
     ('e', 'out0 = 1 / in0'),
     ('f', 'out0 = 1 / in0'),
     ('d', 'out0 = 1 / in0')),
    'out0 = in0 == 0 ? 0 : (1 / in0)',
    doc='''Computes ``1 / x`` elementwise.

    .. seealso:: :data:`numpy.reciprocal`

    ''')


negative = elementwise.create_ufunc(
    'cupy_negative',
    (('?->?', 'out0 = !in0'),
     'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'),
    'out0 = -in0',
    doc='''Takes numerical negative elementwise.

    .. seealso:: :data:`numpy.negative`

    ''')


multiply = ufunc.create_arithmetic(
    'multiply', '*', '&',
    '''Multiplies two arrays elementwise.

    .. seealso:: :data:`numpy.multiply`

    ''')


divide = elementwise.create_ufunc(
    'cupy_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 / in1'),
     ('ff->f', 'out0 = in0 / in1'),
     ('dd->d', 'out0 = in0 / in1')),
    'out0 = in1 == 0 ? 0 : floor((double)in0 / (double)in1)',
    doc='''Divides arguments elementwise.

    .. seealso:: :data:`numpy.divide`

    ''')


power = elementwise.create_ufunc(
    'cupy_power',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = powf(in0, in1)'),
     ('ff->f', 'out0 = powf(in0, in1)'),
     ('dd->d', 'out0 = pow(in0, in1)')),
    'out0 = rint(pow((double)in0, (double)in1))',
    doc='''Computes ``x1 ** x2`` elementwise.

    .. seealso:: :data:`numpy.power`

    ''')


subtract = ufunc.create_arithmetic(
    'subtract', '-', '^',
    '''Subtracts arguments elementwise.

    .. seealso:: :data:`numpy.subtract`

    ''')


true_divide = elementwise.create_ufunc(
    'cupy_true_divide',
    ('bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d', 'll->d', 'LL->d',
     'qq->d', 'QQ->d', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = (out0_type)in0 / (out0_type)in1',
    doc='''Elementwise true division (i.e. division as floating values).

    .. seealso:: :data:`numpy.true_divide`

    ''')


if six.PY3:
    divide = true_divide


floor_divide = elementwise.create_ufunc(
    'cupy_floor_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = _floor_divide(in0, in1)',
    doc='''Elementwise floor division (i.e. integer quotient).

    .. seealso:: :data:`numpy.floor_divide`

    ''')


fmod = elementwise.create_ufunc(
    'cupy_fmod',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = fmodf(in0, in1)'),
     ('ff->f', 'out0 = fmodf(in0, in1)'),
     ('dd->d', 'out0 = fmod(in0, in1)')),
    'out0 = in1 == 0 ? 0 : fmod((double)in0, (double)in1)',
    doc='''Computes the remainder of C division elementwise.

    .. seealso:: :data:`numpy.fmod`

    ''')


modf = elementwise.create_ufunc(
    'cupy_modf',
    ('e->ee', 'f->ff',
     ('d->dd', 'double iptr; out0 = modf(in0, &iptr); out1 = iptr')),
    'float iptr; out0 = modff(in0, &iptr); out1 = iptr',
    doc='''Extracts the fractional and integral parts of an array elementwise.

    This ufunc returns two arrays.

    .. seealso:: :data:`numpy.modf`

    ''')


remainder = elementwise.create_ufunc(
    'cupy_remainder',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 - _floor_divide(in0, in1) * in1)'),
     ('ff->f', 'out0 = in0 - _floor_divide(in0, in1) * in1)'),
     ('dd->d', 'out0 = in0 - _floor_divide(in0, in1) * in1)')),
    'out0 = (in0 - _floor_divide(in0, in1) * in1) * (in1 != 0)',
    doc='''Computes the remainder of Python division elementwise.

    .. seealso:: :data:`numpy.remainder`

    ''')
