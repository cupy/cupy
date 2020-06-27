import scipy

import cupy
import cupyx

from cupy import core
from cupy.core import _routines_math as _math
from cupy.core import fusion
from cupyx.scipy import fft

dot_kernel = core.ReductionKernel(
    'T x1, T x2',
    'T y',
    'x1 * x2',
    'a + b',
    'y = a',
    '0',
    'dot_product'
)


def convolve(a, v, mode='full'):
    """Returns the discrete, linear convolution of two one-dimensional sequences.

    Args:
        a (cupy.ndarray): first 1-dimensional input.
        v (cupy.ndarray): second 1-dimensional input.
        mode (str, optional): `valid`, `same`, `full`

    Returns:
        cupy.ndarray: Discrete, linear convolution of a and v.

    .. seealso:: :func:`numpy.convolve`

    """
    if a.ndim == 0:
        a = a.ravel()
    if v.ndim == 0:
        v = v.ravel()
    if v.size > a.size:
        a, v = v, a
    if a.size == 0:
        raise ValueError('a cannot be empty')
    if v.size == 0:
        raise ValueError('v cannot be empty')
    if v.ndim > 1:
        raise ValueError('v cannot be multidimensional array')

    method = cupyx.scipy.signal.choose_conv_method(a, v, mode)
    if method == 'direct':
        _, out = _dot_convolve(a, v[::-1], mode)
    elif method == 'fft':
        out = _fftconvolve1d(a, v, mode)
    else:
        raise ValueError('Unsupported method')
    return out


def _fftconvolve1d(a1, a2, mode='full'):

    inverted = 0
    if a2.size > a1.size:
        a1, a2 = a2, a1
        inverted = not a2.size % 2
    siz1 = a1.size
    siz2 = a2.size
    shape = siz1 + siz2 - 1
    complex = a1.dtype.kind == 'c' or a2.dtype.kind == 'c'
    # TODO: Replace with cupyx.scipy.fft.next_fast_len
    fshape = scipy.fft.next_fast_len(shape, not complex)
    if not complex:
        fa1 = fft.rfft(a1, fshape)
        fa2 = fft.rfft(a2, fshape)
        out = fft.irfft(fa1 * fa2, fshape)
    else:
        fa1 = fft.fft(a1, fshape)
        fa2 = fft.fft(a2, fshape)
        out = fft.ifft(fa1 * fa2, fshape)
    out = out[slice(shape)]
    if mode == 'same':
        start = (out.size - siz1) / 2 + inverted
        end = start + siz1
        out = out[start: end]
    elif mode == 'valid':
        newsize = siz1 - siz2 + 1
        start = (out.size - newsize) / 2
        end = start + newsize
        out = out[start: end]
    else:
        if mode != 'full':
            raise ValueError('acceptable mode flags are `valid`,'
                             ' `same`, or `full`.')
    result_type = cupy.result_type(a1, a2)
    if result_type.kind in {'u', 'i'}:
        out = cupy.around(out)
    return out.astype(result_type, copy=False)


def _dot_convolve(a1, a2, mode):
    inverted = 0
    dtype = cupy.result_type(a1, a2)
    if a1.size == 0 or a2.size == 0:
        raise ValueError('Array arguments cannot be empty')
    if a1.size < a2.size:
        a1, a2 = a2, a1
        inverted = 1
    length = n1 = a1.size
    n2 = a2.size
    left, right, length = _generate_boundaries(mode, length, n2)
    output = cupy.zeros(length, dtype)
    a1 = a1.astype(dtype, copy=False)
    a2 = a2.astype(dtype, copy=False)
    if left > 0:
        a1_2dims, a2_2dims = _rolling_window(a1, a2, n2 - 1, 'left', left)
        dot_kernel(a1_2dims, a2_2dims, output[:left], axis=1)
    a1_2dims = _rolling_window(a1, a2, n2, 'mid')
    dot_kernel(a1_2dims, a2, output[left:(left + n1 - n2 + 1)], axis=1)
    if right > 0:
        a1_2dims, a2_2dims = _rolling_window(a2, a1[n1 - n2 + 1:],
                                             n2 - 1, 'right', right)
        dot_kernel(a1_2dims, a2_2dims, output[left + n1 - n2 + 1:], axis=1)
    return inverted, output


def _generate_boundaries(mode, length, n):
    if mode == 'valid':
        length += 1 - n
        left = right = 0
    elif mode == 'same':
        left = int(n / 2)
        right = n - left - 1
    elif mode == 'full':
        left = right = n - 1
        length += n - 1
    else:
        raise ValueError('Invalid mode')
    return left, right, length


def _rolling_window(a, b, window, pos, padsize=0):
    if padsize > 0:
        a = cupy.pad(a, padsize)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    a = cupy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if pos == 'mid':
        return a

    a = a[1: 1 + padsize]
    rows, cols = cupy.ogrid[:a.shape[0], :a.shape[1]]
    r = cupy.arange(- a.shape[0] + 1, 1)
    r[r < 0] += a.shape[1]
    cols = cols - r[:, cupy.newaxis]
    a = a[rows, cols]

    b = cupy.pad(b, padsize)
    shape = b.shape[:-1] + (b.shape[-1] - window + 1, window)
    strides = b.strides + (b.strides[-1],)
    b = cupy.lib.stride_tricks.as_strided(b, shape=shape, strides=strides)
    b = b[::-1][1: 1 + padsize]

    if pos == 'left':
        return a, b
    if pos == 'right':
        return a[::-1], b[::-1]
    raise ValueError('Invalid pos')


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


_float_preamble = '''
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
'''
_float_maximum = ('out0 = (isnan(in0) | isnan(in1)) ? out0_type(NAN) : '
                  'out0_type(max(in0, in1))')
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
    preamble=_float_preamble,
    doc='''Takes the maximum of two arrays elementwise.

    If NaN appears, it returns the NaN.

    .. seealso:: :data:`numpy.maximum`

    ''')


_float_minimum = ('out0 = (isnan(in0) | isnan(in1)) ? out0_type(NAN) : '
                  'out0_type(min(in0, in1))')
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
    preamble=_float_preamble,
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
