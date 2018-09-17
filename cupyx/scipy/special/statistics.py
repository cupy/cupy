from cupy import core


ndtr = core.create_ufunc(
    'cupyx_scipy_ndtr', ('f->f', 'd->d'),
    'out0 = normcdf(in0)',
    doc='''Cumulative distribution function of normal distribution.

    .. seealso:: :meth:`scipy.special.ndtr`

    ''')
