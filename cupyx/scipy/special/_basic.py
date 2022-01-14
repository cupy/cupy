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
