from cupy import reduction

sum = reduction.create_reduction_func(
    'cupy_sum',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('a + b', 'in[i]', 'a'), 0)


prod = reduction.create_reduction_func(
    'cupy_prod',
    ['?->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    ('a * b', 'in[i]', 'a'), 1)


def nansum(a, axis=None, dtype=None, out=None, keepdims=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cumprod(a, axis=None, dtype=None, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cumsum(a, axis=None, dtype=None, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def diff(a, n=1, axis=-1, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def ediff1d(ary, to_end=None, to_begin=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def gradient(f, *varargs, **kwargs):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def trapz(y, x=None, dx=1.0, axis=-1):
    # TODO(beam2d): Implement it
    raise NotImplementedError
