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
    '''
    if (in0 < -1) {
        out0 = -1.0 / 0.0;
    } else if (in0 > 1) {
        out0 = 1.0 / 0.0;
    } else {
        out0 = erfinv(in0);
    }
    ''',
    doc='''Inverse function of error function.

    .. seealso:: :meth:`scipy.special.erfinv`

    ''')


erfcinv = core.create_ufunc(
    'cupyx_scipy_erfcinv', ('f->f', 'd->d'),
    '''
    if (in0 < 0) {
        out0 = 1.0 / 0.0;
    } else if (in0 > 2) {
        out0 = -1.0 / 0.0;
    } else {
        out0 = erfcinv(in0);
    }
    ''',
    doc='''Inverse function of complementary error function.

    .. seealso:: :meth:`scipy.special.erfcinv`

    ''')
