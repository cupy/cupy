import numpy

import cupy
from cupy._core._kernel import create_ufunc
from cupy._core._reduction import create_reduction_func
from cupy._util import bf16_loop

from cupy._core.core cimport _ndarray_base


cdef _ndarray_base _ndarray_all(_ndarray_base self, axis, out, keepdims):
    return _all(self, axis=axis, out=out, keepdims=keepdims)


cdef _ndarray_base _ndarray_any(_ndarray_base self, axis, out, keepdims):
    return _any(self, axis=axis, out=out, keepdims=keepdims)


cdef _ndarray_base _ndarray_greater(_ndarray_base self, other):
    return _greater(self, other)


cdef _ndarray_base _ndarray_greater_equal(_ndarray_base self, other):
    return _greater_equal(self, other)


cdef _ndarray_base _ndarray_less(_ndarray_base self, other):
    return _less(self, other)


cdef _ndarray_base _ndarray_less_equal(_ndarray_base self, other):
    return _less_equal(self, other)


cdef _ndarray_base _ndarray_equal(_ndarray_base self, other):
    return _equal(self, other)


cdef _ndarray_base _ndarray_not_equal(_ndarray_base self, other):
    return _not_equal(self, other)


cdef _all = create_reduction_func(
    'cupy_all',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?', 'F->?', 'D->?'),
    ('in0 != type_in0_raw(0)', 'a & b', 'out0 = a', 'bool'),
    'true', '')


cdef _any = create_reduction_func(
    'cupy_any',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?', 'F->?', 'D->?'),
    ('in0 != type_in0_raw(0)', 'a | b', 'out0 = a', 'bool'),
    'false', '')


def promote_weak_int(in_types, weaks):
    if weaks is None:
        return in_types, weaks

    # Python integers can be originally discovered as uint or int.
    # E.g. for int32(-2**31) == 2**32 we need to promote, rather than try
    # to use the NEP 50 style and use int32 (failing the conversion).
    # For float this is currently not what NumPy does (debatable).
    # For bool, NumPy uses the default integer while we make it work (ignore).
    if weaks[0] is int and weaks[1] is False and in_types[1].kind in "iu":
        return in_types, (False, weaks[1])
    elif weaks[1] is int and weaks[0] is False and in_types[0].kind in "iu":
        return in_types, (weaks[0], False)

    return in_types, weaks


def struct_compare_resolution(op, in_dtypes, out_dtypes):
    # NumPy ignores field names, so we'll do that as well.
    dt1, dt2 = in_dtypes
    if dt1.fields is None and dt2.fields is None:
        if dt1.itemsize == dt2.itemsize and dt1.itemsize > 0:
            out_dtypes = (numpy.dtype(bool),)
            return (dt1, dt1), out_dtypes
        raise TypeError(
            f"cannot compare unstructured voids of different size "
            f"({dt1.itemsize} vs {dt2.itemsize})")
    if dt1.fields is None or dt2.fields is None:
        raise TypeError("Cannot compare structured and non-structured dtypes")

    try:
        cmp_dtype = numpy.promote_types(dt1, dt2)
    except TypeError:
        raise TypeError(
            "Cannot compare structured arrays unless they have a common "
            "dtype.  I.e. `np.result_type(arr1, arr2)` must be defined.")

    # Ensure the comparison dtype actually has sufficient alignment.
    cmp_dtype = cupy.make_aligned_dtype(cmp_dtype)

    out_dtypes = (numpy.dtype(bool),)
    return (cmp_dtype, cmp_dtype), out_dtypes


cpdef create_comparison(
        name, op, doc='', no_complex_dtype=True, allow_structured=False):
    ops = (
        '??->?',
        'qq->?', 'QQ->?',
        ('qQ->?', f'out0 = in0 < 0 ? in0 {op} 0 : in0 {op} in1'),
        ('Qq->?', f'out0 = in1 < 0 ? 0 {op} in1 : in0 {op} in1'),
        'ee->?', *bf16_loop(2, '?'), 'ff->?', 'dd->?')

    if not no_complex_dtype:
        ops += ('FF->?', 'DD->?')

    if allow_structured:
        ops += (("VV->?", None, struct_compare_resolution),)

    return create_ufunc(
        'cupy_' + name,
        ops,
        'out0 = in0 %s in1' % op,
        doc=doc, promote_types=promote_weak_int)


cdef _greater = create_comparison(
    'greater', '>',
    '''Tests elementwise if ``x1 > x2``.

    .. seealso:: :data:`numpy.greater`

    ''',
    no_complex_dtype=False)


cdef _greater_equal = create_comparison(
    'greater_equal', '>=',
    '''Tests elementwise if ``x1 >= x2``.

    .. seealso:: :data:`numpy.greater_equal`

    ''',
    no_complex_dtype=False)


cdef _less = create_comparison(
    'less', '<',
    '''Tests elementwise if ``x1 < x2``.

    .. seealso:: :data:`numpy.less`

    ''',
    no_complex_dtype=False)


cdef _less_equal = create_comparison(
    'less_equal', '<=',
    '''Tests elementwise if ``x1 <= x2``.

    .. seealso:: :data:`numpy.less_equal`

    ''',
    no_complex_dtype=False)


cdef _equal = create_comparison(
    'equal', '==',
    '''Tests elementwise if ``x1 == x2``.

    .. seealso:: :data:`numpy.equal`

    ''',
    no_complex_dtype=False, allow_structured=True)


cdef _not_equal = create_comparison(
    'not_equal', '!=',
    '''Tests elementwise if ``x1 != x2``.

    .. seealso:: :data:`numpy.equal`

    ''',
    no_complex_dtype=False, allow_structured=True)


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)
all = _all
any = _any
greater = _greater
greater_equal = _greater_equal
less = _less
less_equal = _less_equal
equal = _equal
not_equal = _not_equal
