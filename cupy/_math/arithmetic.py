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


positive = _core.positive


negative = _core.negative


conjugate = _core.conjugate


# cupy.real is not a ufunc because it returns a view.
# The ufunc implementation is used by fusion.
_real_ufunc = _core.create_ufunc(
    'cupy_real',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = in0.real()'),
     ('D->d', 'out0 = in0.real()')),
    'out0 = in0',
    doc='''Returns the real part of the elements of the array.

    .. seealso:: :func:`numpy.real`

    ''')


# cupy.imag is not a ufunc because it may return a view.
# The ufunc implementation is used by fusion.
_imag_ufunc = _core.create_ufunc(
    'cupy_imag',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = in0.imag()'),
     ('D->d', 'out0 = in0.imag()')),
    'out0 = 0',
    doc='''Returns the imaginary part of the elements of the array.

    .. seealso:: :func:`numpy.imag`

    ''')


def angle(z, deg=False):
    '''Returns the angle of the complex argument.

    .. seealso:: :func:`numpy.angle`

    '''
    if deg:
        return _core.angle_deg(z)
    return _core.angle(z)


def real(val):
    '''Returns the real part of the elements of the array.

    .. seealso:: :func:`numpy.real`

    '''
    if fusion._is_fusing():
        return fusion._call_ufunc(_real_ufunc, val)
    if not isinstance(val, _core.ndarray):
        val = _core.array(val)
    return val.real


def imag(val):
    '''Returns the imaginary part of the elements of the array.

    .. seealso:: :func:`numpy.imag`

    '''
    if fusion._is_fusing():
        return fusion._call_ufunc(_imag_ufunc, val)
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

float_power = _core.create_ufunc(
    'cupy_float_power',
    ('dd->d', 'FF->D',
     ('DD->D', 'out0 = in1 == in1_type(0) ? in1_type(1): pow(in0, in1)')),
    'out0 = pow(in0, in1)',
    doc='''First array elements raised to powers from second array, element-wise.

    .. seealso:: :data:`numpy.float_power`

    '''
)

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
