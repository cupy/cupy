from cupy import core
from cupy.core import _routines_math as _math
from cupy.core import fusion

# TODO(okuta): Implement convolve


def clip(a, a_min=None, a_max=None, out=None):
    """Clips the values of an array to a given interval.

    This is equivalent to ``maximum(minimum(a, a_max), a_min)``, while this
    function is more efficient.

    Args:
        a (cupy.ndarray): The source array.
        a_min (scalar, cupy.ndarray or None): The left side of the interval.
            When it is ``None``, it is ignored.
        a_max (scalar, cupy.ndarray or None): The right side of the interval.
            When it is ``None``, it is ignored.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Clipped array.

    .. seealso:: :func:`numpy.clip`

    """
    if fusion._is_fusing():
        return fusion._call_ufunc(_math.clip,
                                  a, a_min, a_max, out=out)

    # TODO(okuta): check type
    return a.clip(a_min, a_max, out=out)


# sqrt_fixed is deprecated.
# numpy.sqrt is fixed in numpy 1.11.2.
sqrt = sqrt_fixed = core.sqrt


cbrt = core.create_ufunc(
    'cupy_cbrt',
    ('e->e', 'f->f', 'd->d'),
    'out0 = cbrt(in0)',
    doc='''Elementwise cube root function.

    .. seealso:: :data:`numpy.cbrt`

    ''')


square = core.create_ufunc(
    'cupy_square',
    ('b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L', 'q->q',
     'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = in0 * in0',
    doc='''Elementwise square function.

    .. seealso:: :data:`numpy.square`

    ''')


absolute = core.absolute


# TODO(beam2d): Implement it
# fabs


_unsigned_sign = 'out0 = in0 > 0'
_complex_sign = '''
if (in0.real() == 0) {
  out0 = (in0.imag() > 0) - (in0.imag() < 0);
} else {
  out0 = (in0.real() > 0) - (in0.real() < 0);
}
'''
sign = core.create_ufunc(
    'cupy_sign',
    ('b->b', ('B->B', _unsigned_sign), 'h->h', ('H->H', _unsigned_sign),
     'i->i', ('I->I', _unsigned_sign), 'l->l', ('L->L', _unsigned_sign),
     'q->q', ('Q->Q', _unsigned_sign), 'e->e', 'f->f', 'd->d',
     ('F->F', _complex_sign), ('D->D', _complex_sign)),
    'out0 = (in0 > 0) - (in0 < 0)',
    doc='''Elementwise sign function.

    It returns -1, 0, or 1 depending on the sign of the input.

    .. seealso:: :data:`numpy.sign`

    ''')


_float_maximum = 'out0 = isnan(in0) ? in0 : isnan(in1) ? in1 : max(in0, in1)'
maximum = core.create_ufunc(
    'cupy_maximum',
    ('??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', _float_maximum),
     ('ff->f', _float_maximum),
     ('dd->d', _float_maximum),
     ('FF->F', _float_maximum),
     ('DD->D', _float_maximum)),
    'out0 = max(in0, in1)',
    doc='''Takes the maximum of two arrays elementwise.

    If NaN appears, it returns the NaN.

    .. seealso:: :data:`numpy.maximum`

    ''')


_float_minimum = 'out0 = isnan(in0) ? in0 : isnan(in1) ? in1 : min(in0, in1)'
minimum = core.create_ufunc(
    'cupy_minimum',
    ('??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', _float_minimum),
     ('ff->f', _float_minimum),
     ('dd->d', _float_minimum),
     ('FF->F', _float_minimum),
     ('DD->D', _float_minimum)),
    'out0 = min(in0, in1)',
    doc='''Takes the minimum of two arrays elementwise.

    If NaN appears, it returns the NaN.

    .. seealso:: :data:`numpy.minimum`

    ''')


fmax = core.create_ufunc(
    'cupy_fmax',
    ('??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = fmax(in0, in1)'),
     ('ff->f', 'out0 = fmax(in0, in1)'),
     ('dd->d', 'out0 = fmax(in0, in1)'),
     'FF->F', 'DD->D'),
    'out0 = max(in0, in1)',
    doc='''Takes the maximum of two arrays elementwise.

    If NaN appears, it returns the other operand.

    .. seealso:: :data:`numpy.fmax`

    ''')


fmin = core.create_ufunc(
    'cupy_fmin',
    ('??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = fmin(in0, in1)'),
     ('ff->f', 'out0 = fmin(in0, in1)'),
     ('dd->d', 'out0 = fmin(in0, in1)'),
     'FF->F', 'DD->D'),
    'out0 = min(in0, in1)',
    doc='''Takes the minimum of two arrays elementwise.

    If NaN appears, it returns the other operand.

    .. seealso:: :data:`numpy.fmin`

    ''')


_nan_to_num_preamble = '''
template <class T>
__device__ T nan_to_num(T x, T large) {
    if (isnan(x))
        return 0;
    if (isinf(x))
        return copysign(large, x);
    return x;
}

template <class T>
__device__ complex<T> nan_to_num(complex<T> x, T large) {
    T re = nan_to_num(x.real(), large);
    T im = nan_to_num(x.imag(), large);
    return complex<T>(re, im);
}
'''


nan_to_num = core.create_ufunc(
    'cupy_nan_to_num',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H',
     'i->i', 'I->I', 'l->l', 'L->L', 'q->q', 'Q->Q',
     ('e->e',
      'out0 = nan_to_num(in0, float16(32 * 0x7FF))'),
     ('f->f',
      'out0 = nan_to_num(in0, __int_as_float(0x7F800000 - 1))'),
     ('d->d',
      'out0 = nan_to_num(in0, __longlong_as_double(0x7FF0000000000000 - 1))'),
     ('F->F',
      'out0 = nan_to_num(in0, __int_as_float(0x7F800000 - 1))'),
     ('D->D',
      'out0 = nan_to_num(in0, __longlong_as_double(0x7FF0000000000000 - 1))')),
    'out0 = in0',
    preamble=_nan_to_num_preamble,
    doc='''Elementwise nan_to_num function.

    .. seealso:: :data:`numpy.nan_to_num`

    ''')


# TODO(okuta): Implement real_if_close


# TODO(okuta): Implement interp
