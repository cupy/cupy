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
    # Python integers can be originally discovered as uint or int.
    # For comparison we define loops for both so take whatever it is.
    if weaks is None:
        return in_types, weaks

    return in_types, tuple([w if w is not int else False for w in weaks])


cpdef create_comparison(name, op, doc='', no_complex_dtype=True):
    ops = (
        '??->?',
        'qq->?', 'QQ->?',
        ('qQ->?', f'out0 = in0 < 0 ? in0 {op} 0 : in0 {op} in1'),
        ('Qq->?', f'out0 = in1 < 0 ? 0 {op} in1 : in0 {op} in1'),
        'ee->?', *bf16_loop(2, '?'), 'ff->?', 'dd->?')

    if not no_complex_dtype:
        ops += ('FF->?', 'DD->?')

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
    no_complex_dtype=False)


cdef _not_equal = create_comparison(
    'not_equal', '!=',
    '''Tests elementwise if ``x1 != x2``.

    .. seealso:: :data:`numpy.equal`

    ''',
    no_complex_dtype=False)


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
