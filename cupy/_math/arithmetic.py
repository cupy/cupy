from cupy import _core
from cupy._core import fusion


add = _core.add


reciprocal = _core.create_ufunc(
    'cupy_reciprocal',
    ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q',
     ('e', 'out0 = 1 / in0'),
     ('f', 'out0 = 1 / in0'),
     ('d', 'out0 = 1 / in0'),
     ('F', 'out0 = in0_type(1) / in0'),
     ('D', 'out0 = in0_type(1) / in0')),
    'out0 = in0 == 0 ? 0 : (1 / in0)',
    doc='''Computes ``1 / x`` elementwise.

    .. seealso:: :data:`numpy.reciprocal`

    ''')


negative = _core.negative


conjugate = _core.conjugate


angle = _core.angle


def real(val):
    '''Returns the real part of the elements of the array.

    .. seealso:: :func:`numpy.real`

    '''
    if fusion._is_fusing():
        return fusion._call_ufunc(_core.real, val)
    if not isinstance(val, _core.ndarray):
        val = _core.array(val)
    return val.real


def imag(val):
    '''Returns the imaginary part of the elements of the array.

    .. seealso:: :func:`numpy.imag`

    '''
    if fusion._is_fusing():
        return fusion._call_ufunc(_core.imag, val)
    if not isinstance(val, _core.ndarray):
        val = _core.array(val)
    return val.imag


multiply = _core.multiply


divide = _core.divide


divmod = _core.divmod


power = _core.power


subtract = _core.subtract


true_divide = _core.true_divide


floor_divide = _core.floor_divide


fmod = _core.create_ufunc(
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


modf = _core.create_ufunc(
    'cupy_modf',
    ('e->ee', 'f->ff',
     ('d->dd', 'double iptr; out0 = modf(in0, &iptr); out1 = iptr')),
    'float iptr; out0 = modff(in0, &iptr); out1 = iptr',
    doc='''Extracts the fractional and integral parts of an array elementwise.

    This ufunc returns two arrays.

    .. seealso:: :data:`numpy.modf`

    ''')


remainder = _core.remainder
