from cupy import reduction


def all(a, axis=None, out=None, keepdims=False):
    return _all(a, axis=axis, out=out, keepdims=keepdims)


def any(a, axis=None, out=None, keepdims=False):
    return _any(a, axis=axis, out=out, keepdims=keepdims)


_all = reduction.create_reduction_func(
    'cupy_all',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?'),
    ('in0', 'a & b', 'out0 = a', 'bool'),
    'true', '')


_any = reduction.create_reduction_func(
    'cupy_any',
    ('?->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'e->?', 'f->?', 'd->?'),
    ('in0', 'a | b', 'out0 = a', 'bool'),
    'false', '')
