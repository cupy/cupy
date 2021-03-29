import cupy
import cupyx.scipy.fft

from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks


_dot_kernel = _core.ReductionKernel(
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
    else:
        fft, ifft = cupy.fft.rfft, cupy.fft.irfft

    dtype = cupy.result_type(a1, a2)
    n1, n2 = a1.size, a2.size
    out_size = cupyx.scipy.fft.next_fast_len(n1 + n2 - 1)
    fa1 = fft(a1, out_size)
    fa2 = fft(a2, out_size)
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

    if dtype.kind in 'iu':
        out = cupy.around(out)

    return out.astype(dtype, copy=False)


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
sqrt = sqrt_fixed = _core.sqrt


cbrt = _core.create_ufunc(
    'cupy_cbrt',
    ('e->e', 'f->f', 'd->d'),
    'out0 = cbrt(in0)',
    doc='''Elementwise cube root function.

    .. seealso:: :data:`numpy.cbrt`

    ''')


square = _core.create_ufunc(
    'cupy_square',
    ('b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L', 'q->q',
     'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = in0 * in0',
    doc='''Elementwise square function.

    .. seealso:: :data:`numpy.square`

    ''')


absolute = _core.absolute


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
sign = _core.create_ufunc(
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
maximum = _core.create_ufunc(
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
minimum = _core.create_ufunc(
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


fmax = _core.create_ufunc(
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


fmin = _core.create_ufunc(
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


nan_to_num = _core.create_ufunc(
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


@cupy._util.memoize(for_each_device=True)
def _get_interp_kernel(is_complex):
    in_params = 'raw V x, raw U idx, '
    in_params += 'raw W fx, raw Y fy, U len, raw Y left, raw Y right'
    out_params = 'Z y'  # output dtype follows NumPy's

    if is_complex:
        preamble = 'typedef double real_t;\n'
    else:
        preamble = 'typedef Z real_t;\n'
    preamble += 'typedef Z value_t;\n'
    preamble += cupy._sorting.search._preamble  # for _isnan

    code = r'''
        U x_idx = idx[i] - 1;

        if ( _isnan<V>(x[i]) ) { y = x[i]; }
        else if (x_idx < 0) { y = left[0]; }
        else if (x[i] == fx[len - 1]) {
            // searchsorted cannot handle both of the boundary points,
            // so we must detect and correct ourselves...
            y = fy[len - 1];
        }
        else if (x_idx >= len - 1) { y = right[0]; }
        else {
            const Z slope = (value_t)(fy[x_idx+1] - fy[x_idx]) / \
                            ((real_t)fx[x_idx+1] - (real_t)fx[x_idx]);
            Z out = slope * ((real_t)x[i] - (real_t)fx[x_idx]) \
                    + (value_t)fy[x_idx];
            if (_isnan<Z>(out)) {
                out = slope * ((real_t)x[i] - (real_t)fx[x_idx+1]) \
                      + (value_t)fy[x_idx+1];
                if (_isnan<Z>(out) && (fy[x_idx] == fy[x_idx+1])) {
                    out = fy[x_idx];
                }
            }
            y = out;
        }
    '''
    return cupy.ElementwiseKernel(
        in_params, out_params, code, 'cupy_interp', preamble=preamble)


def interp(x, xp, fp, left=None, right=None, period=None):
    """ One-dimensional linear interpolation.

    Args:
        x (cupy.ndarray): a 1D array of points on which the interpolation
            is performed.
        xp (cupy.ndarray): a 1D array of points on which the function values
            (``fp``) are known.
        fp (cupy.ndarray): a 1D array containing the function values at the
            the points ``xp``.
        left (float or complex): value to return if ``x < xp[0]``. Default is
            ``fp[0]``.
        right (float or complex): value to return if ``x > xp[-1]``. Default is
            ``fp[-1]``.
        period (None or float): a period for the x-coordinates. Parameters
            ``left`` and ``right`` are ignored if ``period`` is specified.
            Default is ``None``.

    Returns:
        cupy.ndarray: The interpolated values, same shape as ``x``.

    .. note::
        This function may synchronize if ``left`` or ``right`` is not already
        on the device.

    .. seealso:: :func:`numpy.interp`

    """

    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError('xp and fp must be 1D arrays')
    if xp.size != fp.size:
        raise ValueError('fp and xp are not of the same length')
    if xp.size == 0:
        raise ValueError('array of sample points is empty')
    if not x.flags.c_contiguous:
        raise NotImplementedError('Non-C-contiguous x is currently not '
                                  'supported')
    x_dtype = cupy.common_type(x, xp)
    if not cupy.can_cast(x_dtype, cupy.float64):
        raise TypeError('Cannot cast array data from'
                        ' {} to {} according to the rule \'safe\''
                        .format(x_dtype, cupy.float64))

    if period is not None:
        # The handling of "period" below is modified from NumPy's

        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        left = None
        right = None

        x = x.astype(cupy.float64)
        xp = xp.astype(cupy.float64)

        # normalizing periodic boundaries
        x %= period
        xp %= period
        asort_xp = cupy.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = cupy.concatenate((xp[-1:]-period, xp, xp[0:1]+period))
        fp = cupy.concatenate((fp[-1:], fp, fp[0:1]))
        assert xp.flags.c_contiguous
        assert fp.flags.c_contiguous

    # NumPy always returns float64 or complex128, so we upcast all values
    # on the fly in the kernel
    out_dtype = 'D' if fp.dtype.kind == 'c' else 'd'
    output = cupy.empty(x.shape, dtype=out_dtype)
    idx = cupy.searchsorted(xp, x, side='right')
    left = fp[0] if left is None else cupy.array(left, fp.dtype)
    right = fp[-1] if right is None else cupy.array(right, fp.dtype)
    kern = _get_interp_kernel(out_dtype == 'D')
    kern(x, idx, xp, fp, xp.size, left, right, output)
    return output
