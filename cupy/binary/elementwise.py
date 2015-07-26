from cupy import elementwise


def _create_bit_op(name, op, no_bool=False):
    types = [] if no_bool else ['??->?']
    return elementwise.create_ufunc(
        'cupy_' + name,
        types + ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
                 'LL->L', 'qq->q', 'QQ->Q'],
        'out0 = in0 %s in1' % op)


bitwise_and = _create_bit_op('bitwise_and', '&')
bitwise_or = _create_bit_op('bitwise_or', '|')
bitwise_xor = _create_bit_op('bitwise_xor', '^')


invert = elementwise.create_ufunc(
    'cupy_invert',
    [('?->?', 'out0 = !in0'), 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
     'l->l', 'L->L', 'q->q', 'Q->Q'],
    'out0 = ~in0')


left_shift = _create_bit_op('left_shift', '<<', True)
right_shift = _create_bit_op('right_shift', '>>', True)
