import numpy as np

from cupy._core._kernel import create_ufunc, _Op
from cupy._core._reduction import create_reduction_func

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


def _fix_to_sctype(dtype, sctype):
    # This function doesn't do much (or anything) but it could do more complex
    # type resolution if needed allowing comparison between bytes and strings
    # (but NumPy only supports this for == and != returning all False so raise)
    if dtype.type == sctype:
        return dtype

    raise TypeError("Cannot compare bytes and string dtypes")


def _s_cmp_resolver(op, in_dtypes, out_dtypes):
    # Support only U->S and S->U casts right now

    in1_dtype = _fix_to_sctype(in_dtypes[0], op.in_types[0])
    in2_dtype = _fix_to_sctype(in_dtypes[1], op.in_types[1])
    out_dtype = np.dtype("?")

    return (in1_dtype, in2_dtype), (out_dtype,)


cpdef create_comparison(name, op, doc='', no_complex_dtype=True):

    if no_complex_dtype:
        ops = ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
               'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?')
    else:
        ops = ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
               'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?',
               'FF->?', 'DD->?')

    custom_ops=[
        # Note, mixing right now would cast, but we the code can really do
        # without (C++ might optimize that away.)
        _Op(
            (np.bytes_, np.bytes_), (np.bool_,), f'out0 = in0 {op} in1', None,
            _s_cmp_resolver
        ),
        _Op(
            (np.str_, np.str_), (np.bool_,), f'out0 = in0 {op} in1', None,
            _s_cmp_resolver
        )
    ]

    return create_ufunc(
        'cupy_' + name,
        ops,
        'out0 = in0 %s in1' % op,
        doc=doc,
        custom_ops=custom_ops)


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
