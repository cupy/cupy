import cupy
from cupy import _core

spherical_bessel_preamble = """
#include <cupy/math_constants.h>

__device__ double spherical_yn_real(int n, double x) {
    double s, s0, s1;

    if (isnan(x))
        return x;
    if (x < 0) {
        if (n % 2 == 0)
            return -spherical_yn_real(n, -x);
        else
            return spherical_yn_real(n, -x);
    }
    if (isinf(x))
        return 0;
    if (x == 0)
        return -CUDART_INF;

    s0 = -cos(x) / x;
    if (n == 0) {
        return s0;
    }
    s1 = (s0 - sin(x)) / x;
    for (int k = 2; k <= n; ++k) {
        s = (2.0 * k - 1.0) * s1 / x - s0;
        if (isinf(s)) {
            return s;
        }
        s0 = s1;
        s1 = s;
    }

    return s1;
}

__device__ double spherical_yn_d_real(int n, double x) {
    double s, s0, s1;

    if (isnan(x))
        return x;
    if (x < 0) {
        if (n % 2 == 0)
            return -spherical_yn_d_real(n, -x);
        else
            return spherical_yn_d_real(n, -x);
    }
    if (isinf(x))
        return 0;
    if (x == 0)
        return CUDART_INF;

    if (n == 1) {
        return (sin(x) + cos(x) / x) / x;
    }
    s0 = -cos(x) / x;
    s1 = (s0 - sin(x)) / x;
    for (int k = 2; k <= n; ++k) {
        s = (2.0 * k - 1.0) * s1 / x - s0;
        if (isinf(s)) {
            return s;
        }
        s0 = s1;
        s1 = s;
    }

    return s0 - (n + 1.0) * s1 / x;
}
"""

_spherical_yn_real = _core.create_ufunc(
    "cupyx_scipy_spherical_yn_real",
    ("if->d", "id->d"),
    "out0 = out0_type(spherical_yn_real(in0, in1))",
    preamble=spherical_bessel_preamble,
)

_spherical_dyn_real = _core.create_ufunc(
    "cupyx_scipy_spherical_dyn_real",
    ("if->d", "id->d"),
    "out0 = out0_type(spherical_yn_d_real(in0, in1));",
    preamble=spherical_bessel_preamble,
)


def spherical_yn(n, z, derivative=False):
    """Spherical Bessel function of the second kind or its derivative.

    Parameters
    ----------
    n : cupy.ndarray
        Order of the Bessel function.
    z : cupy.ndarray
        Argument of the Bessel function.
        Real-valued input.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    yn : cupy.ndarray

    See Also
    -------
    :func:`scipy.special.spherical_yn`

    """
    if cupy.iscomplexobj(z):
        if derivative:
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        if derivative:
            return _spherical_dyn_real(n, z)
        else:
            return _spherical_yn_real(n, z)
