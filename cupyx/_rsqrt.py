from __future__ import annotations

from cupy._core._kernel import create_ufunc

rsqrt = create_ufunc(
    'cupy_rsqrt',
    ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = rsqrt(in0)',
    doc='''Returns the reciprocal square root.''')
