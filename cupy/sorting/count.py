from cupy import reduction


def count_nonzero(x):
    return int(_count_nonzero(x))

_count_nonzero = reduction.create_reduction_func(
    'cupy_count_nonzero',
    ('?->l', 'B->l', 'h->l', 'H->l', 'i->l', 'I->l', 'l->l', 'L->l',
     'q->l', 'Q->l', 'e->l', 'f->l', 'd->l'),
    ('in0', 'a + b', 'out0 = (a != 0)', None), 0)
