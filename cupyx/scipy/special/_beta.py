"""Beta and log(abs(beta)) functions.

Also the incomplete beta function and its inverse.

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/beta.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/incbet.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/incbi.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
"""

from cupy import _core
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._gammainc import p1evl_definition


beta_preamble = """
// include for CUDART_INF, CUDART_NAN
#include <cupy/math_constants.h>

// defines from /scipy/special/cephes/const.c
// The ones chosen here are the IEEE variants from that file
__constant__ double MAXLOG = 7.09782712893383996732E2;
__constant__ double MINLOG = -7.08396418532264106224E2;
__constant__ double MACHEP = 1.11022302462515654042E-16;  // 2**-53

// defines from npy_math.h
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */

#define MAXGAM 171.624376956302725
#define ASYMP_FACTOR 1e6
"""


lgam_sgn_definition = """


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

__constant__ double LOGPI = 1.14472988584940017414;

#define MAXLGM 2.556348e305


__noinline__ __device__ static double lgam_sgn(double x, int *sign)
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

lbeta_symp_definition = """
/*
 * Asymptotic expansion for  ln(|B(a, b)|) for a > ASYMP_FACTOR*max(|b|, 1).
 */
__noinline__ __device__ double lbeta_asymp(double a, double b, int *sgn)
{
    double r = lgam_sgn(b, sgn);
    r -= b * log(a);

    r += b*(1-b)/(2*a);
    r += b*(1-b)*(1-2*b)/(12*a*a);
    r += - b*b*(1-b)*(1-b)/(12*a*a*a);

    return r;
}
"""


beta_definition = """

__noinline__ __device__ double beta(double, double);


/*
 * Special case for a negative integer argument
 */

__noinline__ __device__ static double beta_negint(int a, double b)
{
    int sgn;
    if (b == (int)b && 1 - a - b > 0) {
        sgn = ((int)b % 2 == 0) ? 1 : -1;
        return sgn * beta(1 - a - b, b);
    }
    else {
        return CUDART_INF;
    }
}

__noinline__ __device__ double beta(double a, double b)
{
    double y;
    int sign = 1;

    if (a <= 0.0) {
        if (a == floor(a)) {
            if (a == (int)a) {
                return beta_negint((int)a, b);
            }
            else {
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
            return sign * CUDART_INF;
        }
        return sign * exp(y);
    }

    y = Gamma(y);
    a = Gamma(a);
    b = Gamma(b);
    if (y == 0.0) {
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
}
"""


lbeta_definition = """

__noinline__ __device__ double lbeta(double, double);


/*
 * Special case for a negative integer argument
 */

__noinline__ __device__ static double lbeta_negint(int a, double b)
{
    double r;
    if (b == (int)b && 1 - a - b > 0) {
        r = lbeta(1 - a - b, b);
        return r;
    }
    else {
        return CUDART_INF;
    }
}

// Natural log of |beta|

__noinline__ __device__ double lbeta(double a, double b)
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
    return log(y);
}

"""


beta = _core.create_ufunc(
    "cupyx_scipy_beta",
    ("ff->f", "dd->d"),
    "out0 = out0_type(beta(in0, in1));",
    preamble=(
        beta_preamble +
        gamma_definition +
        polevl_definition +
        p1evl_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        beta_definition
    ),
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

    See Also
    --------
    :func:`scipy.special.beta`

    """,
)


betaln = _core.create_ufunc(
    "cupyx_scipy_betaln",
    ("ff->f", "dd->d"),
    "out0 = out0_type(lbeta(in0, in1));",
    preamble=(
        beta_preamble +
        gamma_definition +
        polevl_definition +
        p1evl_definition +
        lgam_sgn_definition +
        lbeta_symp_definition +
        lbeta_definition
    ),
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

    See Also
    --------
    :func:`scipy.special.betaln`

    """,
)


incbet_definition = """

__noinline__ __device__ static double incbd(double, double, double);
__noinline__ __device__ static double incbcf(double, double, double);
__noinline__ __device__ static double pseries(double, double, double);

__constant__ double big = 4.503599627370496e15;
__constant__ double biginv = 2.22044604925031308085e-16;

__noinline__ __device__ double incbet(double aa, double bb, double xx)
{
    double a, b, t, x, xc, w, y;
    int flag;

    if (aa <= 0.0 || bb <= 0.0)
    {
        return CUDART_NAN;
    }

    if ((xx <= 0.0) || (xx >= 1.0)) {
        if (xx == 0.0) {
            return 0.0;
        }
        if (xx == 1.0) {
            return 1.0;
        }
        return CUDART_NAN;
    }

    flag = 0;
    if ((bb * xx) <= 1.0 && xx <= 0.95) {
        t = pseries(aa, bb, xx);
        goto done;
    }

    w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. */
    if (xx > (aa / (aa + bb))) {
        flag = 1;
        a = bb;
        b = aa;
        xc = xx;
        x = w;
    } else {
        a = aa;
        b = bb;
        xc = w;
        x = xx;
    }

    if (flag == 1 && (b * x) <= 1.0 && x <= 0.95) {
        t = pseries(a, b, x);
        goto done;
    }

    /* Choose expansion for better convergence. */
    y = x * (a + b - 2.0) - (a - 1.0);
    if (y < 0.0) {
        w = incbcf(a, b, x);
    } else {
        w = incbd(a, b, x) / xc;
    }

    /* Multiply w by the factor
     * a      b   _             _     _
     * x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

    y = a * log(x);
    t = b * log(xc);
    if ((a + b) < MAXGAM && fabs(y) < MAXLOG && fabs(t) < MAXLOG) {
        t = pow(xc, b);
        t *= pow(x, a);
        t /= a;
        t *= w;
        t *= 1.0 / beta(a, b);
        goto done;
    }
    /* Resort to logarithms.  */
    y += t - lbeta(a,b);
    y += log(w / a);
    if (y < MINLOG) {
        t = 0.0;
    } else {
        t = exp(y);
    }

  done:

    if (flag == 1) {
        if (t <= MACHEP) {
            t = 1.0 - MACHEP;
        } else {
            t = 1.0 - t;
        }
    }
    return t;
}


/* Continued fraction expansion #1
 * for incomplete beta integral
 */

__noinline__ __device__ static double incbcf(double a, double b, double x)
{
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, thresh;
    int n;

    k1 = a;
    k2 = a + b;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = b - 1.0;
    k7 = k4;
    k8 = a + 2.0;

    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    ans = 1.0;
    r = 1.0;
    n = 0;
    thresh = 3.0 * MACHEP;
    do {

        xk = -(x * k1 * k2) / (k3 * k4);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (x * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if (qk != 0)
            r = pk / qk;
        if (r != 0) {
            t = fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }

        if (t < thresh) {
            return ans;
        }

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if ((fabs(qk) + fabs(pk)) > big) {
            pkm2 *= biginv;
            pkm1 *= biginv;
            qkm2 *= biginv;
            qkm1 *= biginv;
        }
        if ((fabs(qk) < biginv) || (fabs(pk) < biginv)) {
            pkm2 *= big;
            pkm1 *= big;
            qkm2 *= big;
            qkm1 *= big;
        }
    }
    while (++n < 300);

    return ans;
}



/* Continued fraction expansion #2
 * for incomplete beta integral
 */

__noinline__ __device__ static double incbd(double a, double b, double x)
{
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, z, thresh;
    int n;

    k1 = a;
    k2 = b - 1.0;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = a + b;
    k7 = a + 1.0;;
    k8 = a + 2.0;

    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    z = x / (1.0 - x);
    ans = 1.0;
    r = 1.0;
    n = 0;
    thresh = 3.0 * MACHEP;
    do {

        xk = -(z * k1 * k2) / (k3 * k4);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (z * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if (qk != 0){
            r = pk / qk;
        }
        if (r != 0) {
            t = fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }

        if (t < thresh) {
            return ans;
        }

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if ((fabs(qk) + fabs(pk)) > big) {
            pkm2 *= biginv;
            pkm1 *= biginv;
            qkm2 *= biginv;
            qkm1 *= biginv;
        }
        if ((fabs(qk) < biginv) || (fabs(pk) < biginv)) {
            pkm2 *= big;
            pkm1 *= big;
            qkm2 *= big;
            qkm1 *= big;
        }
    }
    while (++n < 300);

    return ans;
}


/* Power series for incomplete beta integral.
 * Use when b*x is small and x not too close to 1.  */

__noinline__ __device__ static double pseries(double a, double b, double x)
{
    double s, t, u, v, n, t1, z, ai;

    ai = 1.0 / a;
    u = (1.0 - b) * x;
    v = u / (a + 1.0);
    t1 = v;
    t = u;
    n = 2.0;
    s = 0.0;
    z = MACHEP * ai;
    while (fabs(v) > z) {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * log(x);
    if ((a + b) < MAXGAM && fabs(u) < MAXLOG) {
        t = 1.0 / beta(a, b);
    s = s * t * pow(x, a);
    } else {
        t = -lbeta(a,b) + u + log(s);
        if (t < MINLOG) {
            s = 0.0;
        } else {
            s = exp(t);
        }
    }
    return (s);
}


"""

incbet_preamble = (
    beta_preamble +
    gamma_definition +
    polevl_definition +
    p1evl_definition +
    lgam_sgn_definition +
    lbeta_symp_definition +
    beta_definition +
    lbeta_definition +
    incbet_definition
)


betainc = _core.create_ufunc(
    "cupyx_scipy_betainc",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(incbet(in0, in1, in2));",
    preamble=incbet_preamble,
    doc="""Incomplete beta function.

    Parameters
    ----------
    a, b : cupy.ndarray
        Positive, real-valued parameters
    x : cupy.ndarray
        Real-valued such that 0 <= x <= 1, the upper limit of integration.
    out : ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the incomplete beta function

    See Also
    --------
    :func:`scipy.special.betainc`

    """,
)


incbi_definition = """

__noinline__ __device__ double incbi(double aa, double bb, double yy0)
{
    double a, b, y0, d, y, x, x0, x1, lgm, yp, di, dithresh, yl, yh, xt;
    int i, rflg, dir, nflg;

    i = 0;
    if (yy0 <= 0) {
        return 0.0;
    }
    if (yy0 >= 1.0) {
        return 1.0;
    }
    x0 = 0.0;
    yl = 0.0;
    x1 = 1.0;
    yh = 1.0;
    nflg = 0;

    if (aa <= 1.0 || bb <= 1.0) {
        dithresh = 1.0e-6;
        rflg = 0;
        a = aa;
        b = bb;
        y0 = yy0;
        x = a / (a + b);
        y = incbet(a, b, x);
        goto ihalve;
    }
    else {
        dithresh = 1.0e-4;
    }
    /* approximation to inverse function */

    // normcdfinv is the CUDA Math API equivalent of cephes ndtri
    yp = -normcdfinv(yy0);

    if (yy0 > 0.5) {
        rflg = 1;
        a = bb;
        b = aa;
        y0 = 1.0 - yy0;
        yp = -yp;
    }
    else {
        rflg = 0;
        a = aa;
        b = bb;
        y0 = yy0;
    }

    lgm = (yp * yp - 3.0) / 6.0;
    x = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
    d = yp * sqrt(x + lgm) / x
        - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0))
        * (lgm + 5.0 / 6.0 - 2.0 / (3.0 * x));
    d = 2.0 * d;
    if (d < MINLOG) {
        x = 1.0;
        goto under;
    }
    x = a / (a + b * exp(d));
    y = incbet(a, b, x);
    yp = (y - y0) / y0;
    if (fabs(yp) < 0.2) {
        goto newt;
    }

    /* Resort to interval halving if not close enough. */
ihalve:

    dir = 0;
    di = 0.5;
    for (i = 0; i < 100; i++) {
        if (i != 0) {
            x = x0 + di * (x1 - x0);
            if (x == 1.0) {
                x = 1.0 - MACHEP;
            }
            if (x == 0.0) {
                di = 0.5;
                x = x0 + di * (x1 - x0);
                if (x == 0.0)
                    goto under;
            }
            y = incbet(a, b, x);
            yp = (x1 - x0) / (x1 + x0);
            if (fabs(yp) < dithresh) {
                goto newt;
            }
            yp = (y - y0) / y0;
            if (fabs(yp) < dithresh) {
                goto newt;
            }
        }
        if (y < y0) {
            x0 = x;
            yl = y;
            if (dir < 0) {
                dir = 0;
                di = 0.5;
            } else if (dir > 3) {
                di = 1.0 - (1.0 - di) * (1.0 - di);
            } else if (dir > 1) {
                di = 0.5 * di + 0.5;
            } else {
                di = (y0 - y) / (yh - yl);
            }
            dir += 1;
            if (x0 > 0.75) {
                if (rflg == 1) {
                    rflg = 0;
                    a = aa;
                    b = bb;
                    y0 = yy0;
                } else {
                    rflg = 1;
                    a = bb;
                    b = aa;
                    y0 = 1.0 - yy0;
                }
                x = 1.0 - x;
                y = incbet(a, b, x);
                x0 = 0.0;
                yl = 0.0;
                x1 = 1.0;
                yh = 1.0;
                goto ihalve;
            }
        }
        else {
            x1 = x;
            if (rflg == 1 && x1 < MACHEP) {
                x = 0.0;
                goto done;
            }
            yh = y;
            if (dir > 0) {
                dir = 0;
                di = 0.5;
            } else if (dir < -3) {
                di = di * di;
            } else if (dir < -1) {
                di = 0.5 * di;
            } else {
                di = (y - y0) / (yh - yl);
            }
            dir -= 1;
        }
    }
    if (x0 >= 1.0) {
        x = 1.0 - MACHEP;
        goto done;
    }
    if (x <= 0.0) {
under:
        x = 0.0;
        goto done;
    }

newt:

    if (nflg) {
        goto done;
    }
    nflg = 1;
    lgm = lgam(a + b) - lgam(a) - lgam(b);

    for (i = 0; i < 8; i++) {
        /* Compute the function at this point. */
        if (i != 0) {
            y = incbet(a, b, x);
        }
        if (y < yl) {
            x = x0;
            y = yl;
        } else if (y > yh) {
            x = x1;
            y = yh;
        } else if (y < y0) {
            x0 = x;
            yl = y;
        } else {
            x1 = x;
            yh = y;
        }
        if (x == 1.0 || x == 0.0)
            break;
        /* Compute the derivative of the function at this point. */
        d = (a - 1.0) * log(x) + (b - 1.0) * log(1.0 - x) + lgm;
        if (d < MINLOG) {
            goto done;
        }
        if (d > MAXLOG) {
            break;
        }
        d = exp(d);
        /* Compute the step to the next approximation of x. */
        d = (y - y0) / d;
        xt = x - d;
        if (xt <= x0) {
            y = (x - x0) / (x1 - x0);
            xt = x0 + 0.5 * y * (x - x0);
            if (xt <= 0.0) {
                break;
            }
        }
        if (xt >= x1) {
            y = (x1 - x) / (x1 - x0);
            xt = x1 - 0.5 * y * (x1 - x);
            if (xt >= 1.0) {
                break;
            }
        }
        x = xt;
        if (fabs(d / x) < 128.0 * MACHEP) {
            goto done;
        }
    }
    /* Did not converge.  */
    dithresh = 256.0 * MACHEP;
    goto ihalve;

done:

    if (rflg) {
        if (x <= MACHEP)
            x = 1.0 - MACHEP;
        else
            x = 1.0 - x;
    }
    return x;
}

"""

incbi_preamble = incbet_preamble + incbi_definition


betaincinv = _core.create_ufunc(
    "cupyx_scipy_betaincinv",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(incbi(in0, in1, in2));",
    preamble=incbi_preamble,
    doc="""Inverse of the incomplete beta function.

    Parameters
    ----------
    a, b : cupy.ndarray
        Positive, real-valued parameters
    y : cupy.ndarray
        Real-valued input.
    out : ndarray, optional
        Optional output array for the function result

    Returns
    -------
    scalar or ndarray
        Value of the inverse of the incomplete beta function

    See Also
    --------
    :func:`scipy.special.betaincinv`

    """,
)
