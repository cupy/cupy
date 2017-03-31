from cupy import core
from cupy.math import ufunc

# TODO(okuta): Implement around


# TODO(beam2d): Implement it
# round_ = around


rint = ufunc.create_math_ufunc(
    'rint', 1, 'cupy_rint',
    '''Rounds each element of an array to the nearest integer.

    .. seealso:: :data:`numpy.rint`

    ''')


floor = ufunc.create_math_ufunc(
    'floor', 1, 'cupy_floor',
    '''Rounds each element of an array to its floor integer.

    .. seealso:: :data:`numpy.floor`

    ''')


ceil = ufunc.create_math_ufunc(
    'ceil', 1, 'cupy_ceil',
    '''Rounds each element of an array to its ceiling integer.

    .. seealso:: :data:`numpy.ceil`

    ''')


trunc = ufunc.create_math_ufunc(
    'trunc', 1, 'cupy_trunc',
    '''Rounds each element of an array towards zero.

    .. seealso:: :data:`numpy.trunc`

    ''')


fix = core.create_ufunc(
    'cupy_fix', ('e->e', 'f->f', 'd->d'),
    'out0 = (in0 >= 0.0) ? floor(in0): ceil(in0)',
    doc='''If given value x is positive, it return floor(x).
    Else, it return ceil(x).
    .. seealso:: :data:`numpy.fix`

    ''')
