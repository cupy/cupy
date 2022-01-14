from cupy import _core


log1p = _core.create_ufunc(
    "cupyx_scipy_log1p",
    ("f->f", "d->d"),
    "out0 = out0_type(log1p(in0));",
    doc="""Elementwise function for scipy.special.log1p

    Calculates log(1 + x) for use when `x` is near zero.

    Notes
    -----
    This implementation currently does not support complex-valued `x`.

    .. seealso:: :meth:`scipy.special.log1p`

    """,
)

cbrt = _core.create_ufunc(
    'cupyx_scipy_special_cbrt', ('f->f', 'd->d'),
    'out0 = cbrt(in0)',
    doc='''Cube root.

    .. seealso:: :meth:`scipy.special.cbrt`

    ''')


exp2 = _core.create_ufunc(
    'cupyx_scipy_special_exp2', ('f->f', 'd->d'),
    'out0 = exp2(in0)',
    doc='''Computes ``2**x``.

    .. seealso:: :meth:`scipy.special.exp2`

    ''')


exp10 = _core.create_ufunc(
    'cupyx_scipy_special_exp10', ('f->f', 'd->d'),
    'out0 = exp10(in0)',
    doc='''Computes ``10**x``.

    .. seealso:: :meth:`scipy.special.exp10`

    ''')


expm1 = _core.create_ufunc(
    'cupyx_scipy_special_expm1', ('f->f', 'd->d'),
    'out0 = expm1(in0)',
    doc='''Computes ``exp(x) - 1``.

    .. seealso:: :meth:`scipy.special.expm1`

    ''')
