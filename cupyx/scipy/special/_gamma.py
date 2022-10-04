from cupy import _core
from cupyx.scipy.special._loggamma import loggamma_definition


_gamma_body = """

    if (isinf(in0) && in0 < 0) {
        out0 = -1.0 / 0.0;
    } else if (in0 < 0. && in0 == floor(in0)) {
        out0 = 1.0 / 0.0;
    } else {
        out0 = tgamma(in0);
    }
"""

# Also define a standalone Gamma device function for internal use in other code
# like beta, betaln, etc.
gamma_definition = f"""

__noinline__ __device__ double Gamma(double in0)
{{
    double out0;
    {_gamma_body}
    return out0;
}}
"""

cgamma_definition = loggamma_definition + """

// Compute Gamma(z) using loggamma.

__device__ complex<double> cgamma(complex<double> z)
{
    if ((z.real() <= 0.0) && (z == floor(z.real()))){
        // Poles
        return complex<double>(CUDART_NAN, CUDART_NAN);
    }
    return exp(loggamma(z));
}
"""


gamma = _core.create_ufunc(
    'cupyx_scipy_gamma',
    (
        'f->f',
        'd->d',
        ('F->F', 'out0 = out0_type(cgamma(in0))'),
        ('D->D', 'out0 = cgamma(in0)')
    ),
    _gamma_body,
    preamble=cgamma_definition,
    doc="""Gamma function.

    Args:
        z (cupy.ndarray): The input of gamma function.

    Returns:
        cupy.ndarray: Computed value of gamma function.

    .. seealso:: :func:`scipy.special.gamma`

    """)

# Kernel fusion involves preambles concatenating so
# if there are several kernels that depend on the same cpp function,
# compiler throws an error because of duplicates
# ifndef allows to fixes it as compiler throw all duplicates away
chbevl_implementation = """
#ifndef chbevl_defined
#define chbevl_defined
template<typename T>
__device__ T chbevl(T x, T array[], int n)
{
    T b0, b1, b2, *p;
    int i;

    p = array;
    b0 = *p++;
    b1 = 0.0;
    i = n - 1;

    do {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + *p++;
    }
    while (--i);

    return (0.5 * (b0 - b2));
}
#endif

"""


rgamma_implementation = chbevl_implementation + """

#include <cupy/math_constants.h>

#define MAXLOG 7.09782712893383996732E2
#define NPY_PI 3.141592653589793238462643383279502884


/* Chebyshev coefficients for reciprocal Gamma function
 * in interval 0 to 1.  Function is 1/(x Gamma(x)) - 1
 */

__device__ double R[] = {
    3.13173458231230000000E-17,
    -6.70718606477908000000E-16,
    2.20039078172259550000E-15,
    2.47691630348254132600E-13,
    -6.60074100411295197440E-12,
    5.13850186324226978840E-11,
    1.08965386454418662084E-9,
    -3.33964630686836942556E-8,
    2.68975996440595483619E-7,
    2.96001177518801696639E-6,
    -8.04814124978471142852E-5,
    4.16609138709688864714E-4,
    5.06579864028608725080E-3,
    -6.41925436109158228810E-2,
    -4.98558728684003594785E-3,
    1.27546015610523951063E-1
};


/*
 *     Reciprocal Gamma function
 */
__device__ double rgamma(double x)
{
    double w, y, z;
    int sign;

    if (x > 34.84425627277176174) {
        return exp(-lgamma(x));
    }
    if (x < -34.034) {
        w = -x;
        z = sinpi(w);
        if (z == 0.0) {
            return 0.0;
        }
        if (z < 0.0) {
            sign = 1;
            z = -z;
        } else {
            sign = -1;
        }
        y = log(w * z) - log(NPY_PI) + lgamma(w);
        if (y < -MAXLOG) {
            return sign * 0.0;
        }
        if (y > MAXLOG) {
            return sign * CUDART_INF;
        }
        return sign * exp(y);
    }
    z = 1.0;
    w = x;
    while (w > 1.0) {       /* Downward recurrence */
        w -= 1.0;
        z *= w;
    }
    while (w < 0.0) {       /* Upward recurrence */
        z /= w;
        w += 1.0;
    }
    if (w == 0.0) {     /* Nonpositive integer */
        return 0.0;
    }
    if (w == 1.0) {      /* Other integer */
        return 1.0 / z;
    }
    y = w * (1.0 + chbevl(4.0 * w - 2.0, R, 16)) / z;
    return y;
}

"""


crgamma_implementation = loggamma_definition + """

// Compute 1/Gamma(z) using loggamma

__device__ complex<double> crgamma(complex<double> z)
{

    if ((z.real() <= 0) && (z == floor(z.real()))){
        // Zeros at 0, -1, -2, ...
        return complex<double>(0.0, 0.0);
    }
    return exp(-loggamma(z));  // complex exp via Thrust
}
"""


rgamma = _core.create_ufunc(
    'cupyx_scipy_rgamma',
    (
        'f->f',
        'd->d',
        ('F->F', 'out0 = out0_type(crgamma(in0))'),
        ('D->D', 'out0 = crgamma(in0)')
    ),
    'out0 = out0_type(rgamma(in0))',
    preamble=rgamma_implementation + crgamma_implementation,
    doc="""Reciprocal gamma function.

    Args:
        z (cupy.ndarray): The input to the rgamma function.

    Returns:
        cupy.ndarray: Computed value of the rgamma function.

    .. seealso:: :func:`scipy.special.rgamma`

    """)
