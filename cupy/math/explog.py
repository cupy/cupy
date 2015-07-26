from cupy import elementwise
from cupy.math import ufunc

exp = ufunc.create_math_ufunc('exp', 1)
expm1 = ufunc.create_math_ufunc('expm1', 1)


exp2 = elementwise.create_ufunc(
    'cupy_exp2',
    ['e->e', 'f->f', ('d->d', 'out0 = pow(2., in0)')],
    'out0 = powf(2.f, in0)')


log = ufunc.create_math_ufunc('log', 1)
log10 = ufunc.create_math_ufunc('log10', 1)
log2 = ufunc.create_math_ufunc('log2', 1)
log1p = ufunc.create_math_ufunc('log1p', 1)


logaddexp = elementwise.create_ufunc(
    'cupy_logaddexp',
    ['ee->e', 'ff->f', 'dd->d'],
    'out0 = fmax(in0, in1) + log1p(exp(-fabs(in0 - in1)))')


logaddexp2 = elementwise.create_ufunc(
    'cupy_logaddexp2',
    ['ee->e', 'ff->f', 'dd->d'],
    'out0 = fmax(in0, in1) + log2(1 + exp2(-fabs(in0 - in1)))')
