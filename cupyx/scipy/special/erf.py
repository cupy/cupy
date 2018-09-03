from cupy import core
import cupy.core.fusion


_erf = core.create_ufunc(
    'cupyx_scipy_erf', ('f->f', 'd->d'),
    'out0 = erf(in0)',
    doc='''Error function.

    .. seealso:: :meth:`scipy.special.erf`

    ''')


_erfc = core.create_ufunc(
    'cupyx_scipy_erfc', ('f->f', 'd->d'),
    'out0 = erfc(in0)',
    doc='''Complementary error function.

    .. seealso:: :meth:`scipy.special.erfc`

    ''')


_erfcx = core.create_ufunc(
    'cupyx_scipy_erfcx', ('f->f', 'd->d'),
    'out0 = erfcx(in0)',
    doc='''Scaled complementary error function.

    .. seealso:: :meth:`scipy.special.erfcx`

    ''')


erf = cupy.core.fusion.ufunc(_erf)
erfc = cupy.core.fusion.ufunc(_erfc)
erfcx = cupy.core.fusion.ufunc(_erfcx)
