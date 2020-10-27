from cupy import core


erf = core.create_ufunc(
    'cupyx_scipy_erf', ('f->f', 'd->d'),
    'out0 = erf(in0)',
    doc='''Error function.

    .. seealso:: :meth:`scipy.special.erf`

    ''')


erfc = core.create_ufunc(
    'cupyx_scipy_erfc', ('f->f', 'd->d'),
    'out0 = erfc(in0)',
    doc='''Complementary error function.

    .. seealso:: :meth:`scipy.special.erfc`

    ''')


erfcx = core.create_ufunc(
    'cupyx_scipy_erfcx', ('f->f', 'd->d'),
    'out0 = erfcx(in0)',
    doc='''Scaled complementary error function.

    .. seealso:: :meth:`scipy.special.erfcx`

    ''')


erfinv = core.create_ufunc(
    'cupyx_scipy_erfinv', ('f->f', 'd->d'),
    'out0 = erfinv(in0);',
    doc='''Inverse function of error function.

    .. seealso:: :meth:`scipy.special.erfinv`

    .. note::
        The behavior close to (and outside) the domain follows that of
        SciPy v1.4.0+.

    ''')


erfcinv = core.create_ufunc(
    'cupyx_scipy_erfcinv', ('f->f', 'd->d'),
    'out0 = erfcinv(in0);',
    doc='''Inverse function of complementary error function.

    .. seealso:: :meth:`scipy.special.erfcinv`

    .. note::
        The behavior close to (and outside) the domain follows that of
        SciPy v1.4.0+.

    ''')
