from cupy._core._kernel import create_ufunc
from cupy._core._reduction import create_reduction_func


from cupy._core.core cimport _ndarray_base

cdef _ndarray_base _ascend_all_any(_ndarray_base a, axis, out, keepdims,
    bint is_all):
    # ASCEND: aclnnAll/aclnnAny are broken on bool out -> int32 promote
    # float-cast path -> ViewShape overlap, so use count_nonzero
    import cupy
    import numpy

    nz = cupy.count_nonzero(a != 0, axis=axis)
    if keepdims:
        if axis is not None:
            axes = axis if isinstance(axis, tuple) else (axis,)
            for ax in axes:
                nz = cupy.expand_dims(nz, ax)
        else:
            # axis is None + keepdims: numpy keeps ALL dims as size-1
            # e.g. (1,3, 4) -> (1, 1, 1)
            for _ in range(a.ndim):
                nz = cupy.expand_dims(nz, 0)
    if is_all:
        if axis is None:
            reduced_size = a.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            reduced_size = 1
            for ax in axes:
                reduced_size *= a.shape[ax]
        # all: every reduced element is non-zero
        res = (nz == reduced_size)
    else:
        # any: at least one reduced element is non-zero
        res = (nz > 0)

    if out is not None:
        out[...] = res
        return out
    return res


cdef _ndarray_base _ndarray_all(_ndarray_base self, axis, out, keepdims):
    from cupy.backends.backend import is_ascend
    if is_ascend:
        return _ascend_all_any(self, axis, out, keepdims, True)
    else:
        return _all(self, axis=axis, out=out, keepdims=keepdims)


cdef _ndarray_base _ndarray_any(_ndarray_base self, axis, out, keepdims):
    from cupy.backends.backend import is_ascend
    if is_ascend:
        return _ascend_all_any(self, axis, out, keepdims, False)
    else:
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


cpdef create_comparison(name, op, doc='', no_complex_dtype=True):

    if no_complex_dtype:
        ops = ('??->?', 'qq->?', 'qQ->?', 'Qq->?', 'QQ->?',
               'ee->?', 'ff->?', 'dd->?')
    else:
        ops = ('??->?', 'qq->?', 'qQ->?', 'Qq->?', 'QQ->?',
               'ee->?', 'ff->?', 'dd->?',
               'FF->?', 'DD->?')

    return create_ufunc(
        'cupy_' + name,
        ops,
        'out0 = in0 %s in1' % op,
        doc=doc)


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
