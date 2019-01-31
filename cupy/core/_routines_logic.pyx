from cupy.core._kernel import create_reduction_func

from cupy.core.core cimport ndarray


cdef ndarray _ndarray_all(ndarray self, axis, out, keepdims):
    return _all(self, axis=axis, out=out, keepdims=keepdims)


cdef ndarray _ndarray_any(ndarray self, axis, out, keepdims):
    return _any(self, axis=axis, out=out, keepdims=keepdims)


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


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)


all = _all
any = _any
