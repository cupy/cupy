# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/main/scipy/special/cephes/expn.c
#
#
# Cephes Math Library Release 1.1:  March, 1985
# Copyright 1985 by Stephen L. Moshier
# Direct inquiries to 30 Frost Street, Cambridge, MA 02140

from cupy import _core

from cupyx.scipy.special._gamma import gamma_definition


polevl_definition = '''

__device__ double polevl(double x, double coef[], int N)
{
    double ans;
    double *p;
    p = coef;
    ans = *p++;
    for (int i = 0; i < N; ++i){
        ans = ans * x + *p++;
    }
    return ans;
}

'''


expn_large_n_definition = '''

__constant__ double EUL = 0.57721566490153286060;
__constant__ double BIG = 1.44115188075855872E+17;
__constant__ double MACHEP = 1.11022302462515654042E-16;
__constant__ double MAXLOG = 7.08396418532264106224E2;

#define nA 13

__constant__ double _A0[] = {
    1.00000000000000000
};

__constant__ double _A1[] = {
    1.00000000000000000
};

__constant__ double _A2[] = {
    -2.00000000000000000,
    1.00000000000000000
};

__constant__ double _A3[] = {
    6.00000000000000000,
    -8.00000000000000000,
    1.00000000000000000
};

__constant__ double _A4[] = {
    -24.0000000000000000,
    58.0000000000000000,
    -22.0000000000000000,
    1.00000000000000000
};

__constant__ double _A5[] = {
    120.000000000000000,
    -444.000000000000000,
    328.000000000000000,
    -52.0000000000000000,
    1.00000000000000000
};

__constant__ double _A6[] = {
    -720.000000000000000,
    3708.00000000000000,
    -4400.00000000000000,
    1452.00000000000000,
    -114.000000000000000,
    1.00000000000000000
};

__constant__ double _A7[] = {
    5040.00000000000000,
    -33984.0000000000000,
    58140.0000000000000,
    -32120.0000000000000,
    5610.00000000000000,
    -240.000000000000000,
    1.00000000000000000
};

__constant__ double _A8[] = {
    -40320.0000000000000,
    341136.000000000000,
    -785304.000000000000,
    644020.000000000000,
    -195800.000000000000,
    19950.0000000000000,
    -494.000000000000000,
    1.00000000000000000
};

__constant__ double _A9[] = {
    362880.000000000000,
    -3733920.00000000000,
    11026296.0000000000,
    -12440064.0000000000,
    5765500.00000000000,
    -1062500.00000000000,
    67260.0000000000000,
    -1004.00000000000000,
    1.00000000000000000
};

__constant__ double _A10[] = {
    -3628800.00000000000,
    44339040.0000000000,
    -162186912.000000000,
    238904904.000000000,
    -155357384.000000000,
    44765000.0000000000,
    -5326160.00000000000,
    218848.000000000000,
    -2026.00000000000000,
    1.00000000000000000
};

__constant__ double _A11[] = {
    39916800.0000000000,
    -568356480.000000000,
    2507481216.00000000,
    -4642163952.00000000,
    4002695088.00000000,
    -1648384304.00000000,
    314369720.000000000,
    -25243904.0000000000,
    695038.000000000000,
    -4072.00000000000000,
    1.00000000000000000
};

__constant__ double _A12[] = {
    -479001600.000000000,
    7827719040.00000000,
    -40788301824.0000000,
    92199790224.0000000,
    -101180433024.000000,
    56041398784.0000000,
    -15548960784.0000000,
    2051482776.00000000,
    -114876376.000000000,
    2170626.00000000000,
    -8166.00000000000000,
    1.00000000000000000
};

__constant__ double *A[] = {
    _A0, _A1, _A2,
    _A3, _A4, _A5,
    _A6, _A7, _A8,
    _A9, _A10, _A11,
    _A12
};

__constant__ int Adegs[] = {
    0, 0, 1,
    2, 3, 4,
    5, 6, 7,
    8, 9, 10,
    11
};

/* Asymptotic expansion for large n, DLMF 8.20(ii) */
__device__ double expn_large_n(int n, double x)
{
    int k;
    double p = n;
    double lambda = x/p;
    double multiplier = 1/p/(lambda + 1)/(lambda + 1);
    double fac = 1;
    double res = 1; /* A[0] = 1 */
    double expfac, term;

    expfac = exp(-lambda*p)/(lambda + 1)/p;
    if (expfac == 0) {
        return 0;
    }

    /* Do the k = 1 term outside the loop since A[1] = 1 */
    fac *= multiplier;
    res += fac;

    for (k = 2; k < nA; k++) {
        fac *= multiplier;
        term = fac*polevl(lambda, A[k], Adegs[k]);
        res += term;
        if (fabs(term) < MACHEP*fabs(res)) {
            break;
        }
    }

    return expfac*res;
}

'''


expn_definition = (
    polevl_definition
    + gamma_definition
    + expn_large_n_definition
    + '''

// include for CUDART_NAN, CUDART_INF
#include <cupy/math_constants.h>

__device__ double expn(int n, double x)
{
    double ans, r, t, yk, xk;
    double pk, pkm1, pkm2, qk, qkm1, qkm2;
    double psi, z;
    int i, k;
    double big = BIG;

    if (isnan(x)) {
        return CUDART_NAN;
    } else if (n < 0 || x < 0) {
        return CUDART_NAN;
    }

    if (x > MAXLOG) {
        return (0.0);
    }

    if (x == 0.0) {
        if (n < 2) {
            return CUDART_INF;
        } else {
            return (1.0 / (n - 1.0));
        }
    }

    if (n == 0) {
        return (exp(-x) / x);
    }

    /* Asymptotic expansion for large n, DLMF 8.20(ii) */
    if (n > 50) {
        ans = expn_large_n(n, x);
        return (ans);
    }

    /* Continued fraction, DLMF 8.19.17 */
    if (x > 1.0) {
        k = 1;
        pkm2 = 1.0;
        qkm2 = x;
        pkm1 = 1.0;
        qkm1 = x + n;
        ans = pkm1 / qkm1;

        do {
            k += 1;
            if (k & 1) {
                yk = 1.0;
                xk = n + (k - 1) / 2;
            } else {
                yk = x;
                xk = k / 2;
            }
            pk = pkm1 * yk + pkm2 * xk;
            qk = qkm1 * yk + qkm2 * xk;
            if (qk != 0) {
                r = pk / qk;
                t = fabs((ans - r) / r);
                ans = r;
            } else {
                t = 1.0;
            }
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;
            if (fabs(pk) > big) {
                pkm2 /= big;
                pkm1 /= big;
                qkm2 /= big;
                qkm1 /= big;
            }
        } while (t > MACHEP);

        ans *= exp(-x);
        return ans;
    }

    /* Power series expansion, DLMF 8.19.8 */
    psi = -EUL - log(x);
    for (i = 1; i < n; i++) {
        psi = psi + 1.0 / i;
    }

    z = -x;
    xk = 0.0;
    yk = 1.0;
    pk = 1.0 - n;
    if (n == 1) {
        ans = 0.0;
    } else {
        ans = 1.0 / pk;
    }
    do {
        xk += 1.0;
        yk *= z / xk;
        pk += 1.0;
        if (pk != 0.0) {
            ans += yk / pk;
        }
        if (ans != 0.0)
            t = fabs(yk / ans);
        else
            t = 1.0;
    } while (t > MACHEP);
    k = xk;
    t = n;
    r = n - 1;
    ans = (pow(z, r) * psi / Gamma(t)) - ans;
    return (ans);
}

'''
)


expn = _core.create_ufunc(
    'cupyx_scipy_special_expn',
    ('ff->f', 'dd->d'),
    'out0 = expn(in0, in1)',
    preamble=expn_definition,
    doc="""Generalized exponential integral En.

    Parameters
    ----------
    n : cupy.ndarray
        Non-negative integers
    x : cupy.ndarray
        Real argument

    Returns
    -------
    y : scalar or cupy.ndarray
        Values of the generalized exponential integral

    See Also
    --------
    :func:`scipy.special.expn`

    """,
)
