from cupy import _core


gamma = _core.create_ufunc(
    'cupyx_scipy_gamma', ('f->f', 'd->d'),
    '''
    if (isinf(in0) && in0 < 0) {
        out0 = -1.0 / 0.0;
    } else if (in0 < 0. && in0 == floor(in0)) {
        out0 = 1.0 / 0.0;
    } else {
        out0 = tgamma(in0);
    }
    ''',
    doc="""Gamma function.

    Args:
        z (cupy.ndarray): The input of gamma function.

    Returns:
        cupy.ndarray: Computed value of gamma function.

    .. seealso:: :data:`scipy.special.gamma`

    """)
