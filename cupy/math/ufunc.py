from cupy import elementwise


def create_arithmetic(name, op, boolop):
    return elementwise.create_ufunc(
        'cupy_' + name,
        [('??->?', 'out0 = in0 %s in1' % boolop),
         'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
         'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'],
        'out0 = in0 %s in1' % op)


def create_math_ufunc(math_name, nargs, name=None):
    assert 1 <= nargs <= 2
    if name is None:
        name = 'cupy_' + math_name
    if nargs == 1:
        return elementwise.create_ufunc(
            name, ['e->e', 'f->f', 'd->d'],
            'out0 = %s(in0)' % math_name)
    else:
        return elementwise.create_ufunc(
            name, ['ee->e', 'ff->f', 'dd->d'],
            'out0 = %s(in0, in1)' % math_name)
