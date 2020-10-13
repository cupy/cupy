from cupy import core


j0 = core.create_ufunc(
    'cupyx_scipy_j0', ('f->f', 'd->d'),
    'out0 = j0(in0)',
    doc='''Bessel function of the first kind of order 0.

    .. seealso:: :meth:`scipy.special.j0`

    ''')


j1 = core.create_ufunc(
    'cupyx_scipy_j1', ('f->f', 'd->d'),
    'out0 = j1(in0)',
    doc='''Bessel function of the first kind of order 1.

    .. seealso:: :meth:`scipy.special.j1`

    ''')


y0 = core.create_ufunc(
    'cupyx_scipy_y0', ('f->f', 'd->d'),
    'out0 = y0(in0)',
    doc='''Bessel function of the second kind of order 0.

    .. seealso:: :meth:`scipy.special.y0`

    ''')


y1 = core.create_ufunc(
    'cupyx_scipy_y1', ('f->f', 'd->d'),
    'out0 = y1(in0)',
    doc='''Bessel function of the second kind of order 1.

    .. seealso:: :meth:`scipy.special.y1`

    ''')


i0 = core.create_ufunc(
    'cupyx_scipy_i0', ('f->f', 'd->d'),
    'out0 = cyl_bessel_i0(in0)',
    doc='''Modified Bessel function of order 0.

    .. seealso:: :meth:`scipy.special.i0`

    ''')


i1 = core.create_ufunc(
    'cupyx_scipy_i1', ('f->f', 'd->d'),
    'out0 = cyl_bessel_i1(in0)',
    doc='''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')
