from cupy import _core


# Note: cast complex<single> to complex<double> or tests fail tolerance
log1p = _core.create_ufunc(
    'cupyx_scipy_log1p',
    (('f->f', 'out0 = log1pf(in0)'),
     ('F->F', 'out0 = out0_type(log1p((complex<double>)in0))'),
     'd->d', 'D->D'),
    'out0 = log1p(in0);',
    doc="""Elementwise function for scipy.special.log1p

    Calculates log(1 + x) for use when `x` is near zero.

    Notes
    -----
    This implementation currently does not support complex-valued `x`.

    .. seealso:: :meth:`scipy.special.log1p`

    """,
)
