from cupy import elementwise


def create_arithmetic(name, op, boolop, doc):
    return elementwise.create_ufunc(
        'cupy_' + name,
        (('??->?', 'out0 = in0 %s in1' % boolop),
         'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
         'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'),
        'out0 = in0 %s in1' % op,
        doc=doc)


def create_math_ufunc(math_name, nargs, name, doc):
    assert 1 <= nargs <= 2
    if nargs == 1:
        return elementwise.create_ufunc(
            name, ('e->e', 'f->f', 'd->d'),
            'out0 = %s(in0)' % math_name, doc=doc)
    else:
        return elementwise.create_ufunc(
            name, ('ee->e', 'ff->f', 'dd->d'),
            'out0 = %s(in0, in1)' % math_name, doc=doc)
