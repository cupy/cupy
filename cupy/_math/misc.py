import contextlib

import cupy
import cupyx.scipy.fft

from cupy import core
from cupy.core import _routines_math as _math
from cupy.core import fusion
from cupy.cuda import cufft
from cupy.fft.fft import _output_dtype
from cupy.lib import stride_tricks


_dot_kernel = core.ReductionKernel(
    'T x1, T x2',
    'T y',
    'x1 * x2',
    'a + b',
    'y = a',
    '0',
    'dot_product'
)


def _choose_conv_method(in1, in2, mode):
    if in1.ndim != 1 or in2.ndim != 1:
        raise NotImplementedError('Only 1d inputs are supported currently')

    if in1.dtype.kind in 'bui' or in2.dtype.kind in 'bui':
        return 'direct'

    if _fftconv_faster(in1, in2, mode):
        return 'fft'

    return 'direct'


def _fftconv_faster(x, h, mode):
    """
    .. seealso:: :func: `scipy.signal.signaltools._fftconv_faster`

    """
    # TODO(Dahlia-Chehata): replace with GPU-based constants.
    return True


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
    if a.size == 0:
        raise ValueError('a cannot be empty')
    if v.size == 0:
        raise ValueError('v cannot be empty')
    if v.ndim > 1:
        raise ValueError('v cannot be multidimensional array')
    if v.size > a.size:
        a, v = v, a
    a = a.ravel()
    v = v.ravel()

    method = _choose_conv_method(a, v, mode)
    if method == 'direct':
        out = _dot_convolve(a, v, mode)
    elif method == 'fft':
        out = _fft_convolve(a, v, mode)
    else:
        raise ValueError('Unsupported method')
    return out


def _fft_convolve(a1, a2, mode):

    offset = 0
    if a1.size < a2.size:
        a1, a2 = a2, a1
        offset = 1 - a2.size % 2

    # if either of them is complex, the dtype after multiplication will also be
    if a1.dtype.kind == 'c' or a2.dtype.kind == 'c':
        fft, ifft = cupy.fft.fft, cupy.fft.ifft
        is_c2c = True
    else:
        fft, ifft = cupy.fft.rfft, cupy.fft.irfft
        is_c2c = False

    # hack to work around NumPy/CuPy FFT dtype incompatibility:
    # CuPy internally converts fp16 to fp32 before doing FFT (whereas Numpy
    # converts both fp16 and fp32 to fp64), so here we do the cast early and
    # explicitly, and make sure a correct cuFFT plan can be generated. After
    # the fft-ifft round trip, we cast the output dtype to the correct one.
    out_dtype = cupy.result_type(a1, a2)
    dtype = _output_dtype(out_dtype, 'C2C' if is_c2c else 'R2C')
    a1 = a1.astype(dtype, copy=False)
    a2 = a2.astype(dtype, copy=False)

    n1, n2 = a1.size, a2.size
    out_size = cupyx.scipy.fft.next_fast_len(n1 + n2 - 1)
    # skip calling get_fft_plan() as we know the args exactly
    if is_c2c:
        fft_t = cufft.CUFFT_C2C if dtype == cupy.complex64 else cufft.CUFFT_Z2Z
        fft_plan = cufft.Plan1d(out_size, fft_t, 1)
        ifft_plan = fft_plan
    else:
        fft_t = cufft.CUFFT_R2C if dtype == cupy.float32 else cufft.CUFFT_D2Z
        fft_plan = cufft.Plan1d(out_size, fft_t, 1)
        # this is a no-op context manager
        # TODO(leofang): use contextlib.nullcontext() for PY37+?
        ifft_plan = contextlib.suppress()
    with fft_plan:
        fa1 = fft(a1, out_size)
        fa2 = fft(a2, out_size)
    with ifft_plan:
        out = ifft(fa1 * fa2, out_size)

    if mode == 'full':
        start, end = 0, n1 + n2 - 1
    elif mode == 'same':
        start = (n2 - 1) // 2 + offset
        end = start + n1
    elif mode == 'valid':
        start, end = n2 - 1, n1
    else:
        raise ValueError(
            'acceptable mode flags are `valid`, `same`, or `full`.')

    out = out[start:end]

    if out.dtype.kind in 'iu':
        out = cupy.around(out)

    return out.astype(out_dtype, copy=False)


def _dot_convolve(a1, a2, mode):

    offset = 0
    if a1.size < a2.size:
        a1, a2 = a2, a1
        offset = 1 - a2.size % 2

    dtype = cupy.result_type(a1, a2)
    n1, n2 = a1.size, a2.size
    a1 = a1.astype(dtype, copy=False)
    a2 = a2.astype(dtype, copy=False)

    if mode == 'full':
        out_size = n1 + n2 - 1
        a1 = cupy.pad(a1, n2 - 1)
    elif mode == 'same':
        out_size = n1
        pad_size = (n2 - 1) // 2 + offset
        a1 = cupy.pad(a1, (n2 - 1 - pad_size, pad_size))
    elif mode == 'valid':
        out_size = n1 - n2 + 1

    stride = a1.strides[0]
    a1 = stride_tricks.as_strided(a1, (out_size, n2), (stride, stride))
    output = _dot_kernel(a1, a2[::-1], axis=1)
    return output


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
