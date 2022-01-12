"""Beta and log beta functions.

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/beta.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
"""

from cupy import _core
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_implementation
from cupyx.scipy.special._gammainc import p1evl_definition


misc_preamble = """
// include for CUDART_INF
#include <cupy/math_constants.h>

// defines from /scipy/special/cephes/const.c
__constant__ double MAXLOG = 7.09782712893383996732E2;
__constant__ double LOGPI = 1.14472988584940017414;

// defines from npy_math.h
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */

#define MAXGAM 171.624376956302725
#define ASYMP_FACTOR 1e6
"""


lgam_sgn_implementation = """


/* A[]: Stirling's formula expansion of log Gamma
 * B[], C[]: log Gamma function between 2 and 3
 */
__constant__ double A[] = {
    8.11614167470508450300E-4,
    -5.95061904284301438324E-4,
    7.93650340457716943945E-4,
    -2.77777777730099687205E-3,
    8.33333333333331927722E-2
};

__constant__ double B[] = {
    -1.37825152569120859100E3,
    -3.88016315134637840924E4,
    -3.31612992738871184744E5,
    -1.16237097492762307383E6,
    -1.72173700820839662146E6,
    -8.53555664245765465627E5
};

__constant__ double C[] = {
    /* 1.00000000000000000000E0, */
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6
};

/* log( sqrt( 2*pi ) ) */
__constant__ double LS2PI = 0.91893853320467274178;

#define MAXLGM 2.556348e305


__device__ static double lgam_sgn(double x, int *sign)
{
    double p, q, u, w, z;
    int i;

    *sign = 1;

    if (isinf(x)) {
        return x;
    }

    if (x < -34.0) {
        q = -x;
        w = lgam_sgn(q, sign);
        p = floor(q);
        if (p == q) {
            //lgsing:
            //  sf_error("lgam", SF_ERROR_SINGULAR, NULL);
            return CUDART_INF;
        }
        i = p;
        if ((i & 1) == 0)
            *sign = -1;
        else
            *sign = 1;
        z = q - p;
        if (z > 0.5) {
            p += 1.0;
            z = p - q;
        }
        z = q * sin(NPY_PI * z);
        if (z == 0.0){
            // goto lgsing;
            return CUDART_INF;
        }
        /*     z = log(NPY_PI) - log( z ) - w; */
        z = LOGPI - log(z) - w;
        return z;
    }

    if (x < 13.0) {
        z = 1.0;
        p = 0.0;
        u = x;
        while (u >= 3.0) {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while (u < 2.0) {
            if (u == 0.0) {
                // goto lgsing;
                return CUDART_INF;
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if (z < 0.0) {
            *sign = -1;
            z = -z;
        }
        else {
            *sign = 1;
        }
        if (u == 2.0) {
            return log(z);
        }
        p -= 2.0;
        x = x + p;
        p = x * polevl<5>(x, B) / p1evl<6>(x, C);
        return (log(z) + p);
    }

    if (x > MAXLGM) {
        return (*sign * CUDART_INF);
    }

    q = (x - 0.5) * log(x) - x + LS2PI;
    if (x > 1.0e8) {
        return q;
    }

    p = 1.0 / (x * x);
    if (x >= 1000.0) {
        q += ((7.9365079365079365079365e-4 * p
               - 2.7777777777777777777778e-3) * p
              + 0.0833333333333333333333) / x;
    } else {
        q += polevl<4>(p, A) / x;
    }
    return q;
}


__device__ static double lgam(double x)
{
    int sign;
    return lgam_sgn(x, &sign);
}

"""

lbeta_symp_implementation = """
/*
 * Asymptotic expansion for  ln(|B(a, b)|) for a > ASYMP_FACTOR*max(|b|, 1).
 */
__device__ static double lbeta_asymp(double a, double b, int *sgn)
{
    double r = lgam_sgn(b, sgn);
    r -= b * log(a);

    r += b*(1-b)/(2*a);
    r += b*(1-b)*(1-2*b)/(12*a*a);
    r += - b*b*(1-b)*(1-b)/(12*a*a*a);

    return r;
}
"""


beta_implementation = (
    misc_preamble
    + gamma_implementation
    + polevl_definition
    + p1evl_definition
    + lgam_sgn_implementation
    + lbeta_symp_implementation
    + """

__device__ double beta(double, double);


/*
 * Special case for a negative integer argument
 */

__device__ static double beta_negint(int a, double b)
{
    int sgn;
    if (b == (int)b && 1 - a - b > 0) {
        sgn = ((int)b % 2 == 0) ? 1 : -1;
        return sgn * beta(1 - a - b, b);
    }
    else {
        // sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
        return CUDART_INF;
    }
}

__device__ double beta(double a, double b)
{
    double y;
    int sign = 1;

    if (a <= 0.0) {
        if (a == floor(a)) {
            if (a == (int)a) {
                return beta_negint((int)a, b);
            }
            else {
                //goto overflow;
                return CUDART_INF;
            }
        }
    }

    if (b <= 0.0) {
        if (b == floor(b)) {
            if (b == (int)b) {
                return beta_negint((int)b, a);
            }
            else {
                // goto overflow;
                return CUDART_INF;
            }
        }
    }

    if (fabs(a) < fabs(b)) {
        y = a; a = b; b = y;
    }

    if (fabs(a) > ASYMP_FACTOR * fabs(b) && a > ASYMP_FACTOR) {
        /* Avoid loss of precision in lgam(a + b) - lgam(a) */
        y = lbeta_asymp(a, b, &sign);
        return sign * exp(y);
    }

    y = a + b;
    if (fabs(y) > MAXGAM || fabs(a) > MAXGAM || fabs(b) > MAXGAM) {
        int sgngam;
        y = lgam_sgn(y, &sgngam);
        sign *= sgngam;     /* keep track of the sign */
        y = lgam_sgn(b, &sgngam) - y;
        sign *= sgngam;
        y = lgam_sgn(a, &sgngam) + y;
        sign *= sgngam;
        if (y > MAXLOG) {
            // goto overflow;
            return sign * CUDART_INF;
        }
        return sign * exp(y);
    }

    y = Gamma(y);
    a = Gamma(a);
    b = Gamma(b);
    if (y == 0.0) {
        // goto overflow
        return sign * CUDART_INF;
    }

    if (fabs(fabs(a) - fabs(y)) > fabs(fabs(b) - fabs(y))) {
        y = b / y;
        y *= a;
    }
    else {
        y = a / y;
        y *= b;
    }

    return (y);

// overflow:
//     sf_error("beta", SF_ERROR_OVERFLOW, NULL);
//     return (sign * CUDART_INF);
}
""")


lbeta_implementation = (
    misc_preamble
    + gamma_implementation
    + polevl_definition
    + p1evl_definition
    + lgam_sgn_implementation
    + lbeta_symp_implementation
    + """

__device__ double lbeta(double, double);


/*
 * Special case for a negative integer argument
 */

__device__ static double lbeta_negint(int a, double b)
{
    double r;
    if (b == (int)b && 1 - a - b > 0) {
        r = lbeta(1 - a - b, b);
        return r;
    }
    else {
        // sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
        return CUDART_INF;
    }
}

// Natural log of |beta|

__device__ double lbeta(double a, double b)
{
    double y;
    int sign;

    sign = 1;

    if (a <= 0.0) {
        if (a == floor(a)) {
            if (a == (int)a) {
                return lbeta_negint((int)a, b);
            }
            else {
                // goto over;
                return CUDART_INF;
            }
        }
    }

    if (b <= 0.0) {
        if (b == floor(b)) {
            if (b == (int)b) {
                return lbeta_negint((int)b, a);
            }
            else {
                // goto over;
                return CUDART_INF;
            }
        }
    }

    if (fabs(a) < fabs(b)) {
        y = a; a = b; b = y;
    }

    if (fabs(a) > ASYMP_FACTOR * fabs(b) && a > ASYMP_FACTOR) {
        /* Avoid loss of precision in lgam(a + b) - lgam(a) */
        y = lbeta_asymp(a, b, &sign);
        return y;
    }

    y = a + b;
    if (fabs(y) > MAXGAM || fabs(a) > MAXGAM || fabs(b) > MAXGAM) {
        int sgngam;
        y = lgam_sgn(y, &sgngam);
        sign *= sgngam;     /* keep track of the sign */
        y = lgam_sgn(b, &sgngam) - y;
        sign *= sgngam;
        y = lgam_sgn(a, &sgngam) + y;
        sign *= sgngam;
        return (y);
    }

    y = Gamma(y);
    a = Gamma(a);
    b = Gamma(b);
    if (y == 0.0) {
        // over:
        // sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
        return (sign * CUDART_INF);
    }

    if (fabs(fabs(a) - fabs(y)) > fabs(fabs(b) - fabs(y))) {
        y = b / y;
        y *= a;
    }
    else {
        y = a / y;
        y *= b;
    }

    if (y < 0) {
    y = -y;
    }

    return (log(y));
}
""")


beta = _core.create_ufunc(
    "cupyx_scipy_beta",
    ("ff->f", "dd->d"),
    "out0 = out0_type(beta(in0, in1));",
    preamble=beta_implementation,
    doc="""Beta function.

    Parameters
    ----------
    a, b : cupy.ndarray
        Real-valued arguments
    out : cupy.ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the beta function

    .. seealso:: :meth:`scipy.special.beta`

    """,
)


betaln = _core.create_ufunc(
    "cupyx_scipy_betaln",
    ("ff->f", "dd->d"),
    "out0 = out0_type(lbeta(in0, in1));",
    preamble=lbeta_implementation,
    doc="""Natural logarithm of absolute value of beta function.

    Computes ``ln(abs(beta(a, b)))``.

    Parameters
    ----------
    a, b : cupy.ndarray
        Real-valued arguments
    out : cupy.ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the natural log of the magnitude of beta.

    .. seealso:: :meth:`scipy.special.betaln`

    """,
)
