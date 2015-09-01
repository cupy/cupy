from cupy import elementwise


def create_comparison(name, op, doc=''):
    return elementwise.create_ufunc(
        'cupy_' + name,
        ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?',
         'll->?', 'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?'),
        'out0 = in0 %s in1' % op,
        doc=doc)
