# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/main/scipy/special/specfun/specfun.f

from cupy import _core  # NOQA
from cupyx.scipy.special._exp1 import exp1_definition
from cupyx.scipy.special._exp1 import math_constants_and_eul


expi_definition = """
template < typename T >
__device__ double expi(T x) {
    T ei = 1;
    T r = 1;

    if (x == 0) {
        return -CUDART_INF;
    } else if (x < 0) {
        return -exp1(-x);
    } else if (x <= 40.0) {
        for (int k = 1; k <= 100; k++) {
            int den = (k + 1) * (k + 1);
            r = r * k * x / den;
            ei += r;
        }

        return EUL + x * ei + log(x);
    }

    for (int k = 1; k <= 40; k++) {
        r = r * k / x;
        ei += r;
    }

    return exp(x) / x * ei;
}
"""

expi = _core.create_ufunc(
    'cupyx_scipy_special_expi',
    ('f->f', 'd->d'),
    'out0 = expi(in0)',
    preamble=math_constants_and_eul + exp1_definition + expi_definition,
    doc="""Exponential integral Ei.

    Parameters
    ----------
    x : cupy.ndarray
        Real argument

    Returns
    -------
    y : scalar or cupy.ndarray
        Values of exponential integral

    See Also
    --------
    :func:`scipy.special.expi`

    """,
)
