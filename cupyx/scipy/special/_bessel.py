from cupy import _core


j0 = _core.create_ufunc(
    'cupyx_scipy_special_j0', ('f->f', 'd->d'),
    'out0 = j0(in0)',
    doc='''Bessel function of the first kind of order 0.

    .. seealso:: :meth:`scipy.special.j0`

    ''')


j1 = _core.create_ufunc(
    'cupyx_scipy_special_j1', ('f->f', 'd->d'),
    'out0 = j1(in0)',
    doc='''Bessel function of the first kind of order 1.

    .. seealso:: :meth:`scipy.special.j1`

    ''')


y0 = _core.create_ufunc(
    'cupyx_scipy_special_y0', ('f->f', 'd->d'),
    'out0 = y0(in0)',
    doc='''Bessel function of the second kind of order 0.

    .. seealso:: :meth:`scipy.special.y0`

    ''')


y1 = _core.create_ufunc(
    'cupyx_scipy_special_y1', ('f->f', 'd->d'),
    'out0 = y1(in0)',
    doc='''Bessel function of the second kind of order 1.

    .. seealso:: :meth:`scipy.special.y1`

    ''')


# Note: oddly, unlike for y0 or y1, SciPy always returns double for yn
# dd->d because SciPy will accept 2.0
yn = _core.create_ufunc(
    'cupyx_scipy_special_yn', ('id->d', 'dd->d'),
    'out0 = yn((int)in0, in1)',
    doc='''Bessel function of the second kind of order n.

    Args:
        n (cupy.ndarray): order (integer)
        x (cupy.ndarray): argument (float)

    Returns:
        cupy.ndarray: The result.

    Notes
    -----
    Unlike SciPy, no warning will be raised on unsafe casting of `order` to
    32-bit integer.

    .. seealso:: :meth:`scipy.special.yn`
    ''')


i0 = _core.create_ufunc(
    'cupyx_scipy_special_i0', ('f->f', 'd->d'),
    'out0 = cyl_bessel_i0(in0)',
    doc='''Modified Bessel function of order 0.

    .. seealso:: :meth:`scipy.special.i0`

    ''')


i0e = _core.create_ufunc(
    'cupyx_scipy_special_i0e', ('f->f', 'd->d'),
    'out0 = exp(-abs(in0)) * cyl_bessel_i0(in0)',
    doc='''Exponentially scaled modified Bessel function of order 0.

    .. seealso:: :meth:`scipy.special.i0e`

    ''')


i1 = _core.create_ufunc(
    'cupyx_scipy_special_i1', ('f->f', 'd->d'),
    'out0 = cyl_bessel_i1(in0)',
    doc='''Modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1`

    ''')


i1e = _core.create_ufunc(
    'cupyx_scipy_special_i1e', ('f->f', 'd->d'),
    'out0 = exp(-abs(in0)) * cyl_bessel_i1(in0)',
    doc='''Exponentially scaled modified Bessel function of order 1.

    .. seealso:: :meth:`scipy.special.i1e`

    ''')
