import cupy.core.fusion
from cupy.math import ufunc

try:
    import scipy.special
    _scipy_available = True
except ImportError:
    _scipy_available = False


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
    '''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i0`

    ''')


_i1 = ufunc.create_math_ufunc(
    'cyl_bessel_i1', 1, 'cupyx_scipy_i1',
    '''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')


if _scipy_available:
    j0 = cupy.core.fusion.ufunc(_j0, _j0, scipy.special.j0)
    j1 = cupy.core.fusion.ufunc(_j1, _j1, scipy.special.j1)
    y0 = cupy.core.fusion.ufunc(_y0, _y0, scipy.special.y0)
    y1 = cupy.core.fusion.ufunc(_y1, _y1, scipy.special.y1)
    i0 = cupy.core.fusion.ufunc(_i0, _i0, scipy.special.i0)
    i1 = cupy.core.fusion.ufunc(_i1, _i1, scipy.special.i1)
else:
    def raise_scipy_import_error(x):
        raise ImportError('No module named \'scipy\'')

    j0 = cupy.core.fusion.ufunc(_j0, _j0, raise_scipy_import_error)
    j1 = cupy.core.fusion.ufunc(_j1, _j1, raise_scipy_import_error)
    y0 = cupy.core.fusion.ufunc(_y0, _y0, raise_scipy_import_error)
    y1 = cupy.core.fusion.ufunc(_y1, _y1, raise_scipy_import_error)
    i0 = cupy.core.fusion.ufunc(_i0, _i0, raise_scipy_import_error)
    i1 = cupy.core.fusion.ufunc(_i1, _i1, raise_scipy_import_error)
