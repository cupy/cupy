import cupy.core.fusion
from cupy.math import ufunc


_j0 = ufunc.create_math_ufunc(
    'j0', 1, 'cupyx_scipy_j0',
    '''Bessel function of the first kind of order 0.

    .. seealso:: :meth:`scipy.special.j0`

    ''')


_j1 = ufunc.create_math_ufunc(
    'j1', 1, 'cupyx_scipy_j1',
    '''Bessel function of the first kind of order 1.

    .. seealso:: :meth:`scipy.special.j1`

    ''')


_y0 = ufunc.create_math_ufunc(
    'y0', 1, 'cupyx_scipy_y0',
    '''Bessel function of the second kind of order 0.

    .. seealso:: :meth:`scipy.special.y0`

    ''')


_y1 = ufunc.create_math_ufunc(
    'y1', 1, 'cupyx_scipy_y1',
    '''Bessel function of the second kind of order 1.

    .. seealso:: :meth:`scipy.special.y1`

    ''')


_i0 = ufunc.create_math_ufunc(
    'cyl_bessel_i0', 1, 'cupyx_scipy_i0',
    '''Modified Bessel function of order 0.

    .. seealso:: :meth:`scipy.special.i0`

    ''')


_i1 = ufunc.create_math_ufunc(
    'cyl_bessel_i1', 1, 'cupyx_scipy_i1',
    '''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')


j0 = cupy.core.fusion.ufunc(_j0)
j1 = cupy.core.fusion.ufunc(_j1)
y0 = cupy.core.fusion.ufunc(_y0)
y1 = cupy.core.fusion.ufunc(_y1)
i0 = cupy.core.fusion.ufunc(_i0)
i1 = cupy.core.fusion.ufunc(_i1)
