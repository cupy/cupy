import cupy.core.fusion
from cupy.math import ufunc


_erf = ufunc.create_math_ufunc(
    'erf', 1, 'cupyx_scipy_erf',
    '''Error function.

    .. seealso:: :meth:`scipy.special.erf`

    ''',
    support_complex=False)


_erfc = ufunc.create_math_ufunc(
    'erfc', 1, 'cupyx_scipy_erfc',
    '''Complementary error function.

    .. seealso:: :meth:`scipy.special.erfc`

    ''',
    support_complex=False)


_erfcx = ufunc.create_math_ufunc(
    'erfcx', 1, 'cupyx_scipy_erfcx',
    '''Scaled complementary error function.

    .. seealso:: :meth:`scipy.special.erfcx`

    ''',
    support_complex=False)


erf = cupy.core.fusion.ufunc(_erf)
erfc = cupy.core.fusion.ufunc(_erfc)
erfcx = cupy.core.fusion.ufunc(_erfcx)
