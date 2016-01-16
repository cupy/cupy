from cupy import core


add = core.add


reciprocal = core.create_ufunc(
    'cupy_reciprocal',
    ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q',
     ('e', 'out0 = 1 / in0'),
     ('f', 'out0 = 1 / in0'),
     ('d', 'out0 = 1 / in0')),
    'out0 = in0 == 0 ? 0 : (1 / in0)',
    doc='''Computes ``1 / x`` elementwise.

    .. seealso:: :data:`numpy.reciprocal`

    ''')


negative = core.negative


multiply = core.multiply


divide = core.divide


power = core.power


subtract = core.subtract


true_divide = core.true_divide


floor_divide = core.floor_divide


fmod = core.create_ufunc(
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


modf = core.create_ufunc(
    'cupy_modf',
    ('e->ee', 'f->ff',
     ('d->dd', 'double iptr; out0 = modf(in0, &iptr); out1 = iptr')),
    'float iptr; out0 = modff(in0, &iptr); out1 = iptr',
    doc='''Extracts the fractional and integral parts of an array elementwise.

    This ufunc returns two arrays.

    .. seealso:: :data:`numpy.modf`

    ''')


remainder = core.remainder
