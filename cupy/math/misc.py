from cupy import elementwise

_id = 'out0 = in0'


def convolve(a, v, mode='full', allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


_clip = elementwise.create_ufunc(
    'cupy_clip',
    ['bbb->b', 'BBB->B', 'hhh->h', 'HHH->H', 'iii->i', 'III->I', 'lll->l',
     'LLL->L', 'qqq->q', 'QQQ->Q', 'eee->e', 'fff->f', 'ddd->d'],
    'out0 = min(in2, max(in1, in0))')


def clip(a, a_min, a_max, out=None, allocator=None):
    return _clip(a, a_min, a_max, out=out, allocator=allocator)


sqrt = elementwise.create_ufunc(
    'cupy_sqrt',
    # I think this order is a bug of NumPy, though we select this "buggy"
    # behavior for compatibility with NumPy.
    ['f->f', 'd->d', 'e->e'],
    'out0 = sqrt(in0)')


# Fixed version of sqrt
sqrt_fixed = elementwise.create_ufunc(
    'cupy_sqrt',
    ['e->e', 'f->f', 'd->d'],
    'out0 = sqrt(in0)')


square = elementwise.create_ufunc(
    'cupy_square',
    ['b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L', 'q->q',
     'Q->Q', 'e->e', 'f->f', 'd->d'],
    'out0 = in0 * in0')


absolute = elementwise.create_ufunc(
    'cupy_absolute',
    [('?->?', _id), 'b->b', ('B->B', _id), 'h->h', ('H->H', _id), 'i->i',
     ('I->I', _id), 'l->l', ('L->L', _id), 'q->q', ('Q->Q', _id),
     ('e->e', 'out0 = fabsf(in0)'),
     ('f->f', 'out0 = fabsf(in0)'),
     ('d->d', 'out0 = fabs(in0)')],
    'out0 = in0 > 0 ? in0 : -in0')


# TODO(beam2d): Implement it
# fabs


_unsigned_sign = 'out0 = in0 > 0'
sign = elementwise.create_ufunc(
    'cupy_sign',
    ['b->b', ('B->B', _unsigned_sign), 'h->h', ('H->H', _unsigned_sign),
     'i->i', ('I->I', _unsigned_sign), 'l->l', ('L->L', _unsigned_sign),
     'q->q', ('Q->Q', _unsigned_sign), 'e->e', 'f->f', 'd->d'],
    'out0 = (in0 > 0) - (in0 < 0)')


_float_maximum = \
    'out0 = isnan(in0) ? in0 : isnan(in1) ? in1 : max(in0, in1)'
maximum = elementwise.create_ufunc(
    'cupy_maximum',
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', _float_maximum),
     ('ff->f', _float_maximum),
     ('dd->d', _float_maximum)],
    'out0 = max(in0, in1)')


_float_minimum = \
    'out0 = isnan(in0) ? in0 : isnan(in1) ? in1 : min(in0, in1)'
minimum = elementwise.create_ufunc(
    'cupy_minimum',
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q',
     ('ee->e', _float_minimum),
     ('ff->f', _float_minimum),
     ('dd->d', _float_minimum)],
    'out0 = min(in0, in1)')


fmax = elementwise.create_ufunc(
    'cupy_fmax',
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'],
    'out0 = max(in0, in1)')


fmin = elementwise.create_ufunc(
    'cupy_fmin',
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'],
    'out0 = min(in0, in1)')


def nan_to_num(x, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def real_if_close(a, tol=100, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def interp(x, xp, fp, left=None, right=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError
