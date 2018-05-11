from cupy.core.fusion import _create_ufunc
from cupy.math import ufunc

import scipy.special


j0 = _create_ufunc(
    ufunc.create_math_ufunc(
        'j0', 1, 'cupyx_scipy_j0',
        '''Bessel function of the first kind of order 0.

        .. seealso:: :meth:`scipy.special.j0`

        '''),
    scipy.special.j0)


j1 = _create_ufunc(
    ufunc.create_math_ufunc(
        'j1', 1, 'cupyx_scipy_j1',
        '''Bessel function of the first kind of order 1.

        .. seealso:: :meth:`scipy.special.j1`

        '''),
    scipy.special.j1)


y0 = _create_ufunc(
    ufunc.create_math_ufunc(
        'y0', 1, 'cupyx_scipy_y0',
        '''Bessel function of the second kind of order 0.

        .. seealso:: :meth:`scipy.special.y0`

        '''),
    scipy.special.y0)


y1 = _create_ufunc(
    ufunc.create_math_ufunc(
        'y1', 1, 'cupyx_scipy_y1',
        '''Bessel function of the second kind of order 1.

        .. seealso:: :meth:`scipy.special.y1`

        '''),
    scipy.special.y1)


i0 = _create_ufunc(
    ufunc.create_math_ufunc(
        'cyl_bessel_i0', 1, 'cupyx_scipy_i0',
        '''Modified Bessel function of order 1.

        .. seealso:: :meth:`scipy.special.i0`

        '''),
    scipy.special.i0)


i1 = _create_ufunc(
    ufunc.create_math_ufunc(
        'cyl_bessel_i1', 1, 'cupyx_scipy_i1',
        '''Modified Bessel function of order 1.

        .. seealso:: :meth:`scipy.special.i1`

        '''),
    scipy.special.i1)
