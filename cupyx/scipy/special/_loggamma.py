"""
The source code here is an adaptation with minimal changes from the following
SciPy files:

https://github.com/scipy/scipy/blob/master/scipy/special/_complexstuff.pxd
https://github.com/scipy/scipy/blob/master/scipy/special/_evalpoly.pxd
https://github.com/scipy/scipy/blob/master/scipy/special/_loggamma.pxd

"""

from cupy import _core
from cupyx.scipy.special._complexstuff import zlog1_definition
from cupyx.scipy.special._trig import csinpi_definition


ceval_poly_definition = """

/* Evaluate polynomials.
 *
 *
 * All of the coefficients are stored in reverse order, i.e. if the
 * polynomial is
 *
 *     u_n x^n + u_{n - 1} x^{n - 1} + ... + u_0,
 *
 * then coeffs[0] = u_n, coeffs[1] = u_{n - 1}, ..., coeffs[n] = u_0.
 *
 * References
 * ----------
 * [1] Knuth, 'The Art of Computer Programming, Volume II'
 *
 */

/* Evaluate a polynomial with real coefficients at a complex point.
 * Uses equation (3) in section 4.6.4 of [1]. Note that it is more
 * efficient than Horner's method.
 */
__device__ complex<double> cevalpoly(double *coeffs, int degree,
                                     complex<double> z)
{
    int j;
    double a = coeffs[0];
    double b = coeffs[1];
    double r = 2*z.real();
    double s = z.real()*z.real() + z.imag()*z.imag();
    double tmp;
    for (j=2; j<=degree; j++)
    {
        tmp = b;
        b = fma(-s, a, coeffs[j]);
        a = fma(r, a, tmp);
    }
    return z*a + b;
}
"""


loggamma_real_definition = """

#include <cupy/math_constants.h>

__device__ double loggamma_real(double x)
{
    if (x < 0.0) {
        return CUDART_NAN;
    } else {
        return lgamma(x);
    }
}

"""


loggamma_definition = (
    ceval_poly_definition
    + zlog1_definition
    + csinpi_definition
    + loggamma_real_definition
    + """

#include <cupy/math_constants.h>

#define TWOPI 6.2831853071795864769252842  // 2*pi
#define LOGPI 1.1447298858494001741434262  // log(pi)
#define HLOG2PI 0.918938533204672742  // log(2*pi)/2
#define SMALLX 7.0
#define SMALLY 7.0
#define TAYLOR_RADIUS 0.2

__device__ complex<double> loggamma_recurrence(complex<double>);
__device__ complex<double> loggamma_stirling(complex<double>);
__device__ complex<double> loggamma_taylor(complex<double>);


// Compute the principal branch of log-Gamma

__noinline__ __device__ complex<double> loggamma(complex<double> z)
{
    double tmp;
    if (isnan(z)) {
        return complex<double>(CUDART_NAN, CUDART_NAN);
    } else if ((z.real() <= 0) && (z == floor(z.real()))) {
        return complex<double>(CUDART_NAN, CUDART_NAN);
    } else if ((z.real() > SMALLX) || (fabs(z.imag()) > SMALLY)) {
        return loggamma_stirling(z);
    } else if (abs(z - 1.0) <= TAYLOR_RADIUS) {
        return loggamma_taylor(z);
    } else if (abs(z - 2.0) <= TAYLOR_RADIUS) {
        // Recurrence relation and the Taylor series around 1
        return zlog1(z - 1.0) + loggamma_taylor(z - 1.0);
    } else if (z.real() < 0.1) {
        // Reflection formula; see Proposition 3.1 in [1]
        tmp = copysign(TWOPI, z.imag())*floor(0.5*z.real() + 0.25);
        complex<double> ctemp(LOGPI, tmp);
        return ctemp - log(csinpi(z)) - loggamma(1.0 - z);
    } else if (signbit(z.imag()) == 0.0) {
        // z.imag() >= 0 but is not -0.0
        return loggamma_recurrence(z);
    } else {
        return conj(loggamma_recurrence(conj(z)));
    }
}


/* Backward recurrence relation.
 *
 * See Proposition 2.2 in [1] and the Julia implementation [2].
 */
__noinline__ __device__ complex<double> loggamma_recurrence(complex<double> z)
{
    int signflips = 0;
    int sb = 0;
    int nsb;
    complex<double> shiftprod = z;
    z += 1.0;
    while(z.real() <= SMALLX) {
        shiftprod *= z;
        nsb = signbit(shiftprod.imag());
        if (nsb != 0 and sb == 0) {
            signflips += 1;
        }
        sb = nsb;
        z += 1.0;
    }
    complex<double> ctemp(0.0, -signflips*TWOPI);
    return loggamma_stirling(z) - log(shiftprod) + ctemp;
}


/* Stirling series for log-Gamma.
 *
 * The coefficients are B[2*n]/(2*n*(2*n - 1)) where B[2*n] is the
 * (2*n)th Bernoulli number. See (1.1) in [1].
 */
__device__ complex<double> loggamma_stirling(complex<double> z)
{
    double coeffs[] = {
        -2.955065359477124183e-2, 6.4102564102564102564e-3,
        -1.9175269175269175269e-3, 8.4175084175084175084e-4,
        -5.952380952380952381e-4, 7.9365079365079365079e-4,
        -2.7777777777777777778e-3, 8.3333333333333333333e-2
    };
    complex<double> rz = 1.0/z;
    complex<double> rzz = rz/z;
    return (z - 0.5)*log(z) - z + HLOG2PI + rz*cevalpoly(coeffs, 7, rzz);
}


/* Taylor series for log-Gamma around z = 1.
 *
 * It is
 *
 * loggamma(z + 1) = -gamma*z + zeta(2)*z**2/2 - zeta(3)*z**3/3 ...
 *
 * where gamma is the Euler-Mascheroni constant.
 */
__device__ complex<double> loggamma_taylor(complex<double> z)
{
    double coeffs[] = {
        -4.3478266053040259361e-2, 4.5454556293204669442e-2,
        -4.7619070330142227991e-2, 5.000004769810169364e-2,
        -5.2631679379616660734e-2, 5.5555767627403611102e-2,
        -5.8823978658684582339e-2, 6.2500955141213040742e-2,
        -6.6668705882420468033e-2, 7.1432946295361336059e-2,
        -7.6932516411352191473e-2, 8.3353840546109004025e-2,
        -9.0954017145829042233e-2, 1.0009945751278180853e-1,
        -1.1133426586956469049e-1, 1.2550966952474304242e-1,
        -1.4404989676884611812e-1, 1.6955717699740818995e-1,
        -2.0738555102867398527e-1, 2.7058080842778454788e-1,
        -4.0068563438653142847e-1, 8.2246703342411321824e-1,
        -5.7721566490153286061e-1
    };

    z = z - 1.0;
    return z*cevalpoly(coeffs, 22, z);
}

""")


loggamma = _core.create_ufunc(
    'cupyx_scipy_loggamma',
    (
        ('f->f', 'out0 = out0_type(loggamma_real(in0))'),
        ('d->d', 'out0 = loggamma_real(in0)'),
        'F->F',
        'D->D'
    ),
    'out0 = out0_type(loggamma(in0))',
    preamble=loggamma_definition,
    doc="""Principal branch of the logarithm of the gamma function.

    Parameters
    ----------
    z : cupy.ndarray
        Values in the complex plain at which to compute loggamma
    out : cupy.ndarray, optional
        Output array for computed values of loggamma

    Returns
    -------
    cupy.ndarray
        Values of loggamma at z.

    See Also
    --------
    :func:`scipy.special.loggamma`
    """,
)
