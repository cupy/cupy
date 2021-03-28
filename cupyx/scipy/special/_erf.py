from cupy import _core


erf = _core.create_ufunc(
    'cupyx_scipy_erf', ('f->f', 'd->d'),
    'out0 = erf(in0)',
    doc='''Error function.

    .. seealso:: :meth:`scipy.special.erf`

    ''')


erfc = _core.create_ufunc(
    'cupyx_scipy_erfc', ('f->f', 'd->d'),
    'out0 = erfc(in0)',
    doc='''Complementary error function.

    .. seealso:: :meth:`scipy.special.erfc`

    ''')


erfcx = _core.create_ufunc(
    'cupyx_scipy_erfcx', ('f->f', 'd->d'),
    'out0 = erfcx(in0)',
    doc='''Scaled complementary error function.

    .. seealso:: :meth:`scipy.special.erfcx`

    ''')


erfinv = _core.create_ufunc(
    'cupyx_scipy_erfinv', ('f->f', 'd->d'),
    'out0 = erfinv(in0);',
    doc='''Inverse function of error function.

    .. seealso:: :meth:`scipy.special.erfinv`

    .. note::
        The behavior close to (and outside) the domain follows that of
        SciPy v1.4.0+.

    ''')


erfcinv = _core.create_ufunc(
    'cupyx_scipy_erfcinv', ('f->f', 'd->d'),
    'out0 = erfcinv(in0);',
    doc='''Inverse function of complementary error function.

    .. seealso:: :meth:`scipy.special.erfcinv`

    .. note::
        The behavior close to (and outside) the domain follows that of
        SciPy v1.4.0+.

    ''')
