import string
import sys

import numpy

import cupy
#from cupy._core._reduction import create_reduction_func # TODO: reduction
from cupy._core._kernel import create_ufunc, _get_warpsize
from cupy._core._scalar import get_typename
from cupy._core._ufuncs import elementwise_copy
import cupy._core.core as core
from cupy._core cimport internal
from cupy import _util

from cupy_backends.cuda.api cimport runtime
from cupy._core cimport _accelerator
from cupy._core._dtype cimport get_dtype
from cupy._core._routines_creation cimport _ndarray_init
#from cupy._core._compile_with_cache cimport compile_with_cache
from cupy._core.core cimport _ndarray_base
from cupy.xpu cimport memory

# TODO: ASCEND not suported
#from cupy.cuda import cub

try:
    import cupy_backends.cuda.libs.cutensor as cuda_cutensor
except ImportError:
    cuda_cutensor = None


# _ndarray_base members


cdef _ndarray_base _ndarray_conj(_ndarray_base self):
    if self.dtype.kind == 'c':
        return _conjugate(self)
    else:
        return self


cdef _ndarray_base _ndarray_real_getter(_ndarray_base self):
    if self.dtype.kind == 'c':
        dtype = get_dtype(self.dtype.char.lower())
        view = core.ndarray.__new__(
            type(self), shape=self._shape, dtype=dtype, _obj=self,
            memptr=self.data, strides=self._strides)
        (<_ndarray_base>view).base = (
            self.base if self.base is not None else self)
        return view
    return self


cdef _ndarray_base _ndarray_real_setter(_ndarray_base self, value):
    elementwise_copy(value, _ndarray_real_getter(self))


cdef _ndarray_base _ndarray_imag_getter(_ndarray_base self):
    cdef memory.MemoryPointer memptr
    if self.dtype.kind == 'c':
        dtype = get_dtype(self.dtype.char.lower())
        memptr = self.data
        # Make the memory pointer point to the first imaginary element.
        # Note that even if the array doesn't have a valid memory (e.g. 0-size
        # array), the resulting array should be a view of the original array,
        # aligning with NumPy behavior.
        if memptr.ptr != 0:
            memptr = memptr + self.dtype.itemsize // 2
        view = core.ndarray.__new__(
            type(self), shape=self._shape, dtype=dtype, memptr=memptr,
            strides=self._strides)
        (<_ndarray_base>view).base = (
            self.base if self.base is not None else self)
        return view
    new_array = core.ndarray.__new__(type(self), self.shape, dtype=self.dtype)
    new_array.fill(0)
    return new_array


cdef _ndarray_base _ndarray_imag_setter(_ndarray_base self, value):
    if self.dtype.kind == 'c':
        elementwise_copy(value, _ndarray_imag_getter(self))
    else:
        raise TypeError('cupy.ndarray does not have imaginary part to set')


cdef _ndarray_base _ndarray_prod(
        _ndarray_base self, axis, dtype, out, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        result = None
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_PROD, axis, dtype, out, keepdims)
        if (accelerator == _accelerator.ACCELERATOR_CUTENSOR and
                cuda_cutensor is not None):
            from cupyx import cutensor
            result = cutensor._try_reduction_routine(
                self, axis, dtype, out, keepdims, cuda_cutensor.OP_MUL, 1, 0)
        if result is not None:
            return result
    if dtype is None:
        return _prod_auto_dtype(self, axis, dtype, out, keepdims)
    else:
        return _prod_keep_dtype(self, axis, dtype, out, keepdims)


cdef _ndarray_base _ndarray_sum(
        _ndarray_base self, axis, dtype, out, keepdims):
    for accelerator in _accelerator._routine_accelerators:
        result = None
        if accelerator == _accelerator.ACCELERATOR_CUB:
            # result will be None if the reduction is not compatible with CUB
            result = cub.cub_reduction(
                self, cub.CUPY_CUB_SUM, axis, dtype, out, keepdims)
        if (accelerator == _accelerator.ACCELERATOR_CUTENSOR and
                cuda_cutensor is not None):
            from cupyx import cutensor
            result = cutensor._try_reduction_routine(
                self, axis, dtype, out, keepdims, cuda_cutensor.OP_ADD, 1, 0)
        if result is not None:
            return result

    if dtype is None:
        return _sum_auto_dtype(self, axis, dtype, out, keepdims)
    else:
        return _sum_keep_dtype(self, axis, dtype, out, keepdims)


cdef _ndarray_base _ndarray_cumsum(_ndarray_base self, axis, dtype, out):
    return cupy.cumsum(self, axis, dtype, out)


cdef _ndarray_base _ndarray_cumprod(_ndarray_base self, axis, dtype, out):
    return cupy.cumprod(self, axis, dtype, out)


cdef _ndarray_base _ndarray_clip(_ndarray_base self, a_min, a_max, out):
    kind = self.dtype.kind
    if a_min is None:
        if kind == 'f':
            a_min = self.dtype.type('-inf')
        elif kind in 'iu':
            a_min = numpy.iinfo(self.dtype.type).min
    if a_max is None:
        if kind == 'f':
            a_max = self.dtype.type('inf')
        elif kind in 'iu':
            a_max = numpy.iinfo(self.dtype.type).max
    return _clip(self, a_min, a_max, out=out)


# private/internal

_op_char = {scan_op.SCAN_SUM: '+', scan_op.SCAN_PROD: '*'}
_identity = {scan_op.SCAN_SUM: 0, scan_op.SCAN_PROD: 1}


cpdef _ndarray_base _nansum(_ndarray_base a, axis, dtype, out, keepdims):
    if cupy.iscomplexobj(a):
        return _nansum_complex_dtype(a, axis, dtype, out, keepdims)
    elif dtype is None:
        return _nansum_auto_dtype(a, axis, dtype, out, keepdims)
    else:
        return _nansum_keep_dtype(a, axis, dtype, out, keepdims)


cpdef _ndarray_base _nanprod(_ndarray_base a, axis, dtype, out, keepdims):
    if cupy.iscomplexobj(a):
        return _nanprod_complex_dtype(a, axis, dtype, out, keepdims)
    elif dtype is None:
        return _nanprod_auto_dtype(a, axis, dtype, out, keepdims)
    else:
        return _nanprod_keep_dtype(a, axis, dtype, out, keepdims)


if sys.platform == "win32":
    _sumprod_types = (
        '?->q', 'b->q', 'B->Q', 'h->q', 'H->Q', 'i->q', 'I->Q', 'l->q', 'L->Q',
        'q->q', 'Q->Q',
        ('e->e', (None, None, None, 'float')),
        'f->f', 'd->d', 'F->F', 'D->D',
    )
else:
    _sumprod_types = (
        '?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
        'q->q', 'Q->Q',
        ('e->e', (None, None, None, 'float')),
        'f->f', 'd->d', 'F->F', 'D->D',
    )


_sum_auto_dtype = create_reduction_func(
    'cupy_sum', _sumprod_types,
    ('in0', 'a + b', 'out0 = type_out0_raw(a)', None), 0)


_sum_keep_dtype = create_reduction_func(
    'cupy_sum_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a + b', 'out0 = type_out0_raw(a)', None), 0)


_nansum_auto_dtype = create_reduction_func(
    'cupy_nansum', _sumprod_types,
    ('(in0 == in0) ? in0 : type_in0_raw(0)',
     'a + b', 'out0 = type_out0_raw(a)', None), 0)


_nansum_keep_dtype = create_reduction_func(
    'cupy_nansum_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('(in0 == in0) ? in0 : type_in0_raw(0)',
     'a + b', 'out0 = type_out0_raw(a)', None), 0)


_nansum_complex_dtype = create_reduction_func(
    'cupy_nansum_complex_dtype',
    ('F->F', 'D->D'),
    ('''
    type_in0_raw((in0.real() == in0.real()) ? in0.real() : 0,
                 (in0.imag() == in0.imag()) ? in0.imag() : 0)
    ''',
     'a + b', 'out0 = type_out0_raw(a)', None), 0)


_prod_auto_dtype = create_reduction_func(
    'cupy_prod', _sumprod_types,
    ('in0', 'a * b', 'out0 = type_out0_raw(a)', None), 1)


_prod_keep_dtype = create_reduction_func(
    'cupy_prod_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('in0', 'a * b', 'out0 = type_out0_raw(a)', None), 1)


_nanprod_auto_dtype = create_reduction_func(
    'cupy_nanprod', _sumprod_types,
    ('(in0 == in0) ? in0 : type_in0_raw(1)',
     'a * b', 'out0 = type_out0_raw(a)', None), 1)


_nanprod_keep_dtype = create_reduction_func(
    'cupy_nanprod_with_dtype',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d', 'F->F', 'D->D'),
    ('(in0 == in0) ? in0 : type_in0_raw(1)',
     'a * b', 'out0 = type_out0_raw(a)', None), 1)


_nanprod_complex_dtype = create_reduction_func(
    'cupy_nanprod_complex_dtype',
    ('F->F', 'D->D'),
    ('''
    type_in0_raw((in0.real() == in0.real()) ? in0.real() : 1,
                 (in0.imag() == in0.imag()) ? in0.imag() : 1)
    ''',
     'a * b', 'out0 = type_out0_raw(a)', None), 1)

cdef create_arithmetic(
        name, op, boolop, doc, cutensor_op=None, scatter_op=None):
    # boolop is either
    #  - str (the operator for bool-bool inputs) or
    #  - callable (a function to raise an error for bool-bool inputs).
    if isinstance(boolop, str):
        boolop = 'out0 = in0 %s in1' % boolop

    return create_ufunc(
        'cupy_' + name,
        (('??->?', boolop),
         'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
         'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'FF->F',
         'DD->D'),
        'out0 = in0 %s in1' % op,
        doc=doc,
        cutensor_op=cutensor_op,
        scatter_op=scatter_op)


_add = create_arithmetic(
    'add', '+', '|',
    '''Adds two arrays elementwise.

    .. seealso:: :data:`numpy.add`

    ''',
    cutensor_op=('OP_ADD', 1, 1), scatter_op='add')


_conjugate = create_ufunc(
    'cupy_conjugate',
    ('b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L', 'q->q',
     'Q->Q', 'e->e', 'f->f', 'd->d',
     ('F->F', 'out0 = conj(in0)'),
     ('D->D', 'out0 = conj(in0)')),
    'out0 = in0',
    doc='''Returns the complex conjugate, element-wise.

    .. seealso:: :data:`numpy.conjugate`

    ''')


_angle = create_ufunc(
    'cupy_angle',
    ('?->d', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = arg(in0)'),
     ('D->d', 'out0 = arg(in0)')),
    'out0 = in0 >= 0 ? 0 : M_PI',
    doc='''Returns the angle of the complex argument.

    .. seealso:: :func:`numpy.angle`

    ''')


_angle_deg = create_ufunc(
    'cupy_angle_deg',
    ('?->d', 'e->e', 'f->f', 'd->d',
     ('F->f', 'out0 = arg(in0) * (180.0 / M_PI)'),
     ('D->d', 'out0 = arg(in0) * (180.0 / M_PI)')),
    'out0 = in0 >= 0 ? 0 : 180.0',
    doc='''Returns the angle of the complex argument.

    .. seealso:: :func:`numpy.angle`

    ''')


def _positive_boolean_error():
    raise TypeError(
        'The cupy boolean positive, the `+` operator, is not supported.')


_positive = create_ufunc(
    'cupy_positive',
    (('?->?', _positive_boolean_error),
     'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = +in0',
    doc='''Takes numerical positive elementwise.

    .. seealso:: :data:`numpy.positive`

    ''')


def _negative_boolean_error():
    raise TypeError(
        'The cupy boolean negative, the `-` operator, is not supported, '
        'use the `~` operator or the logical_not function instead.')


_negative = create_ufunc(
    'cupy_negative',
    (('?->?', _negative_boolean_error),
     'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = -in0',
    doc='''Takes numerical negative elementwise.

    .. seealso:: :data:`numpy.negative`

    ''')


_multiply = create_arithmetic(
    'multiply', '*', '&',
    '''Multiplies two arrays elementwise.

    .. seealso:: :data:`numpy.multiply`

    ''',
    cutensor_op=('OP_MUL', 1, 1))


# `integral_power` should return somewhat appropriate values for negative
# integral powers (for which NumPy would raise errors). Hence the branches in
# the beginning. This behavior is not officially documented and could change.
cdef _power_preamble = '''
template <typename T>
inline __device__ T integral_power(T in0, T in1) {
    if (in1 < 0) {
        if (in0 == -1) {return (in1 & 1) ? -1 : 1;}
        else {return (in0 == 1) ? 1 : 0;}
    }
    T out0 = 1;
    while (in1 > 0) {
        if (in1 & 1) {
            out0 *= in0;
        }
        in0 *= in0;
        in1 >>= 1;
    }
    return out0;
}

template <typename T>
inline __device__ T complex_power(T in0, T in1) {
    return in1 == T(0) ? T(1): pow(in0, in1);
}
'''

_power = create_ufunc(
    'cupy_power',
    ('??->b', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = powf(in0, in1)'),
     ('ff->f', 'out0 = powf(in0, in1)'),
     ('dd->d', 'out0 = pow(in0, in1)'),
     ('FF->F', 'out0 = complex_power(in0, in1)'),
     ('DD->D', 'out0 = complex_power(in0, in1)')),
    'out0 = integral_power(in0, in1)',
    preamble=_power_preamble,
    doc='''Computes ``x1 ** x2`` elementwise.

    .. seealso:: :data:`numpy.power`

    ''')


def _subtract_boolean_error():
    raise TypeError(
        'cupy boolean subtract, the `-` operator, is deprecated, use the '
        'bitwise_xor, the `^` operator, or the logical_xor function instead.')


_subtract = create_arithmetic(
    'subtract', '-', _subtract_boolean_error,
    '''Subtracts arguments elementwise.

    .. seealso:: :data:`numpy.subtract`

    ''',
    cutensor_op=('OP_ADD', 1, -1), scatter_op='sub')


# NB: Cannot define loops with short ints in the NEP 50 world. Consider
# `cupy.arange(3, dtype=cp.uint8) / (-2)`. It would select the 'BB->d' loop,
# and the kernel would have a declaration `uint8_t in1;`, this converts
# -2 to uint8_t at initialization (modulo UINT8_MAX, likely).
# The family of qq loops is a work-around to achieve almost correct promotion.
# TODO(seberg): Per-ufunc promotion or per-loop type resolution is probably
#               needed for a full fix.
_true_divide = create_ufunc(
    'cupy_true_divide',
    ('qq->d', 'qQ->d', 'Qq->d', 'QQ->d',
     'ee->e', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
    'out0 = static_cast<out0_type>(in0) / static_cast<out0_type>(in1)',
    doc='''Elementwise true division (i.e. division as floating values).

    .. seealso:: :data:`numpy.true_divide`

    ''',
    out_ops=('ee->e', 'ff->f', 'dd->d', 'FF->F', 'DD->D'),
)


_divide = _true_divide


_floor_divide = create_ufunc(
    'cupy_floor_divide',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
    'out0 = _floor_divide(in0, in1)',
    doc='''Elementwise floor division (i.e. integer quotient).

    .. seealso:: :data:`numpy.floor_divide`

    ''')


_remainder = create_ufunc(
    'cupy_remainder',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ee->e', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('ff->f', 'out0 = in0 - _floor_divide(in0, in1) * in1'),
     ('dd->d', 'out0 = in0 - _floor_divide(in0, in1) * in1')),
    'out0 = (in0 - _floor_divide(in0, in1) * in1) * (in1 != 0)',
    doc='''Computes the remainder of Python division elementwise.

    .. seealso:: :data:`numpy.remainder`

    ''')


_absolute = create_ufunc(
    'cupy_absolute',
    (('?->?', 'out0 = in0'),
     'b->b', ('B->B', 'out0 = in0'), 'h->h', ('H->H', 'out0 = in0'),
     'i->i', ('I->I', 'out0 = in0'), 'l->l', ('L->L', 'out0 = in0'),
     'q->q', ('Q->Q', 'out0 = in0'),
     ('e->e', 'out0 = fabsf(in0)'),
     ('f->f', 'out0 = fabsf(in0)'),
     ('d->d', 'out0 = fabs(in0)'),
     ('F->f', 'out0 = abs(in0)'),
     ('D->d', 'out0 = abs(in0)')),
    'out0 = in0 > 0 ? in0 : -in0',
    doc='''Elementwise absolute value function.

    .. seealso:: :data:`numpy.absolute`

    ''')


_sqrt = create_ufunc(
    'cupy_sqrt',
    ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = sqrt(in0)',
    doc='''Elementwise square root function.

    .. seealso:: :data:`numpy.sqrt`

    ''')


_clip = create_ufunc(
    'cupy_clip',
    ('???->?', 'bbb->b', 'BBB->B', 'hhh->h', 'HHH->H', 'iii->i', 'III->I',
     'lll->l', 'LLL->L', 'qqq->q', 'QQQ->Q', 'eee->e', 'fff->f', 'ddd->d'),
    'out0 = in1 > in2 ? in2 : (in0 < in1 ? in1 : (in0 > in2 ? in2 : in0))')


# =============================================================================
# Routines: round divmod split from more.pyx
# =============================================================================
cdef str _id = 'out0 = in0'

cdef str _divmod_float = '''
    out0_type a = _floor_divide(in0, in1);
    out0 = a;
    out1 = in0 - a * in1'''


_divmod = create_ufunc(
    'cupy_divmod',
    ('bb->bb', 'BB->BB', 'hh->hh', 'HH->HH', 'ii->ii', 'II->II', 'll->ll',
     'LL->LL', 'qq->qq', 'QQ->QQ',
     ('ee->ee', _divmod_float),
     ('ff->ff', _divmod_float),
     ('dd->dd', _divmod_float)),
    '''
    if (in1 == 0) {
        out0 = 0;
        out1 = 0;
    } else {
        out0_type a = _floor_divide(in0, in1);
        out0 = a;
        out1 = in0 - a * in1;
    }''')


cdef _round_preamble = '''
#ifdef __HIP_DEVICE_COMPILE__
#define round_float llrintf
#else
#define round_float __float2ll_rn
#endif

template<typename T> __device__ T pow10(long long n){
  T x = 1, a = 10;
  while (n) {
    if (n & 1) x *= a;
    a *= a;
    n >>= 1;
  }
  return x;
};
'''


cdef _round_float = '''
if (in1 == 0) {
    out0 = rint(in0);
} else {
    double x;
    x = pow10<double>(abs(in1));  // TODO(okuta): Move before loop
    out0 = in1 < 0 ? rint(in0 / x) * x : rint(in0 * x) / x;
}'''

cdef _round_complex = '''
if (in1 == 0) {
    out0 = in0_type(rint(in0.real()), rint(in0.imag()));
} else {
    double x = pow10<double>(abs(in1));  // TODO(okuta): Move before loop
    if (in1 < 0) {
        out0 = in0_type(rint(in0.real() / x) * x,
                        rint(in0.imag() / x) * x);
    } else {
        out0 = in0_type(rint(in0.real() * x) / x,
                        rint(in0.imag() * x) / x);
    }
}'''


# There is a known incompatibility with NumPy (as of 1.16.4) such as
# `numpy.around(2**63, -1) == cupy.around(2**63, -1)` gives `False`.
#
# NumPy seems to round integral values via double.  As double has
# only 53 bit precision, last few bits of (u)int64 value may be lost.
# As a consequence, `numpy.around(2**63, -1)` does NOT round up the
# last digit (9223372036854775808 instead of ...810).
#
# The following code fixes the problem, so `cupy.around(2**63, -1)`
# gives `...810`, which (may correct but) is incompatible with NumPy.
_round_ufunc = create_ufunc(
    'cupy_round',
    ('?q->e',
     'bq->b', 'Bq->B', 'hq->h', 'Hq->H', 'iq->i', 'Iq->I', 'lq->l', 'Lq->L',
     'qq->q', 'Qq->Q',
     ('eq->e', _round_float),
     ('fq->f', _round_float),
     ('dq->d', _round_float),
     ('Fq->F', _round_complex),
     ('Dq->D', _round_complex)),
    '''
    out0 = in0;
    ''', preamble=_round_preamble)


_round_ufunc_neg_uint = create_ufunc(
    'cupy_round_neg_uint',
    ('?q->e',
     'bq->b', 'Bq->B', 'hq->h', 'Hq->H', 'iq->i', 'Iq->I', 'lq->l', 'Lq->L',
     'qq->q', 'Qq->Q'),
    '''
        // TODO(okuta): Move before loop
        long long x = pow10<long long>(in1 - 1);

        // TODO(okuta): Check Numpy
        // `cupy.around(-123456789, -4)` works as follows:
        // (1) scale by `x` above: -123456.789
        // (2) split at the last 2 digits: -123400 + (-5.6789 * 10)
        // (3) round the latter by `rint()`: -123400 + (-6.0 * 10)
        // (4) unscale by `x` above: -123460000
        long long q = in0 / x / 100;
        int r = in0 - q*x*100;
        out0 = (q*100 + round_float(r/(x*10.0f))*10) * x;
    ''', preamble=_round_preamble)


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)


add = _add
conjugate = _conjugate
angle = _angle
angle_deg = _angle_deg
positive = _positive
negative = _negative
multiply = _multiply
divide = _divide
power = _power
subtract = _subtract
true_divide = _true_divide
floor_divide = _floor_divide
remainder = _remainder
absolute = _absolute
sqrt = _sqrt

sum_auto_dtype = _sum_auto_dtype  # used from cupy/math/sumprod.py
nansum_auto_dtype = _nansum_auto_dtype  # used from cupy/math/sumprod.py
prod_auto_dtype = _prod_auto_dtype  # used from cupy/math/sumprod.py
nanprod_auto_dtype = _nanprod_auto_dtype  # used from cupy/math/sumprod.py
clip = _clip  # used from cupy/math/misc.py
