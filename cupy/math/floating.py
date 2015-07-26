from cupy import elementwise
from cupy.math import ufunc

signbit = elementwise.create_ufunc(
    'cupy_signbit',
    ['e->?', 'f->?', 'd->?'],
    'out0 = signbit(in0)')


copysign = ufunc.create_math_ufunc('copysign', 2)


ldexp = elementwise.create_ufunc(
    'cupy_ldexp',
    ['ei->e', 'fi->f', 'el->e', 'fl->f', 'di->d', 'dl->d'],
    'out0 = ldexp(in0, in1)')


frexp = elementwise.create_ufunc(
    'cupy_frexp',
    ['e->ei', 'f->fi', 'd->di'],
    'int nptr; out0 = frexp(in0, &nptr); out1 = nptr')


nextafter = ufunc.create_math_ufunc('nextafter', 2)
