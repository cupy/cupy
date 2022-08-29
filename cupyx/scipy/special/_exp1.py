# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/main/scipy/special/specfun/specfun.f

from cupy import _core  # NOQA


math_constants_and_eul = """
#include <cupy/math_constants.h>
__constant__ double EUL = 0.5772156649015328;
"""

exp1_definition = """
template <typename T>
static __device__ T exp1(T x) {
    if (x == 0) {
        return CUDART_INF;
    } else if (x <= 1) {
        T e1 = 1.0;
        T R = 1.0;

        for (int k = 1; k <= 25; k++) {
            int den = (k + 1) * (k + 1);
            R = -R * k * x / den;
            e1 += R;
        }
        return -EUL - log(x) + x * e1;
    }

    int M = 20 + 80.0 / x;
    T t0 = 0;
    for (int k = M; k != 0; k--) {
        t0 = k / (1.0 + k / (x + t0));
    }
    T t = 1.0 / (x + t0);
    return exp(-x) * t;
}
"""


exp1 = _core.create_ufunc(
    'cupyx_scipy_special_exp1',
    ('f->f', 'd->d'),
    'out0 = exp1(in0)',
    preamble=math_constants_and_eul + exp1_definition,
    doc="""Exponential integral E1.

    Parameters
    ----------
    x : cupy.ndarray
        Real argument

    Returns
    -------
    y : scalar or cupy.ndarray
        Values of the exponential integral E1

    See Also
    --------
    :func:`scipy.special.exp1`

    """,
)
