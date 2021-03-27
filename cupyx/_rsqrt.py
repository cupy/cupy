from cupy._core.core import create_ufunc

rsqrt = create_ufunc(
    'cupy_rsqrt',
    ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = rsqrt(in0)',
    doc='''Returns the reciprocal square root.''')
