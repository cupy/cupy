"""Beta and log beta functions.

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/ndtri.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
"""

from cupy import _core
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gammainc import p1evl_definition


misc_preamble = """
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


ndtri_definition = """

/* sqrt(2pi) */
__constant__ double s2pi = 2.50662827463100050242E0;

/* approximation for 0 <= |y - 0.5| <= 3/8 */
__constant__ double P0[5] = {
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
};

__constant__ double Q0[8] = {
    /* 1.00000000000000000000E0, */
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
};

/* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
 * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
 */
__constant__ double P1[9] = {
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
};

__constant__ double Q1[8] = {
    /*  1.00000000000000000000E0, */
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
};

/* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
 * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
 */

__constant__ double P2[9] = {
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
};

__constant__ double Q2[8] = {
    /*  1.00000000000000000000E0, */
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
};


__device__ static double ndtri(double y0)
{
    double x, y, z, y2, x0, x1;
    int code;

    if (y0 == 0.0) {
        return -CUDART_INF;
    }
    if (y0 == 1.0) {
        return CUDART_INF;
    }
    if (y0 < 0.0 || y0 > 1.0) {
        // sf_error("ndtri", SF_ERROR_DOMAIN, NULL);
        return CUDART_NAN;
    }
    code = 1;
    y = y0;
    if (y > (1.0 - 0.13533528323661269189)) {   /* 0.135... = exp(-2) */
        y = 1.0 - y;
        code = 0;
    }

    if (y > 0.13533528323661269189) {
        y = y - 0.5;
        y2 = y * y;
        x = y + y * (y2 * polevl<4>(y2, P0) / p1evl<8>(y2, Q0));
        x = x * s2pi;
        return x;
    }

    x = sqrt(-2.0 * log(y));
    x0 = x - log(x) / x;

    z = 1.0 / x;
    if (x < 8.0)        /* y > exp(-32) = 1.2664165549e-14 */
        x1 = z * polevl<8>(z, P1) / p1evl<8>(z, Q1);
    else
        x1 = z * polevl<8>(z, P2) / p1evl<8>(z, Q2);
    x = x0 - x1;
    if (code != 0)
        x = -x;
    return x;
}

"""


ndtri = _core.create_ufunc(
    "cupyx_scipy_ndtri",
    ("f->f", "d->d"),
    "out0 = out0_type(ndtri(in0));",
    preamble=(
        polevl_definition +
        p1evl_definition +
        ndtri_definition
    ),
    doc="""Inverse of Gaussian cumulative distribution function.

    Parameters
    ----------
    y : cupy.ndarray
        Input argument.
    out : cupy.ndarray, optional
        Optional output array for the function result.

    Returns
    -------
    x : scalar or ndarray
        Value of the ndtri function.

    .. seealso:: :meth:`scipy.special.ndtri`

    """,
)
