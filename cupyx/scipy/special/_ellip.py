# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ellipk.c
# https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ellipk.pxd
#
#
# Cephes Math Library Release 2.8:  June, 2000
# Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier

from cupy import _core
from cupyx.scipy.special._digamma import polevl_definition

ellpk_definition = """
#include <cupy/math_constants.h>

__constant__ double P[] = {
    1.37982864606273237150E-4,
    2.28025724005875567385E-3,
    7.97404013220415179367E-3,
    9.85821379021226008714E-3,
    6.87489687449949877925E-3,
    6.18901033637687613229E-3,
    8.79078273952743772254E-3,
    1.49380448916805252718E-2,
    3.08851465246711995998E-2,
    9.65735902811690126535E-2,
    1.38629436111989062502E0
};

__constant__ double Q[] = {
    2.94078955048598507511E-5,
    9.14184723865917226571E-4,
    5.94058303753167793257E-3,
    1.54850516649762399335E-2,
    2.39089602715924892727E-2,
    3.01204715227604046988E-2,
    3.73774314173823228969E-2,
    4.88280347570998239232E-2,
    7.03124996963957469739E-2,
    1.24999999999870820058E-1,
    4.99999999999999999821E-1
};

__constant__ double C1 = 1.3862943611198906188E0;	/* log(4) */


static __device__ double ellpk(double x)
{
    double MACHEP = 1.11022302462515654042E-16;	/* 2**-53 */


    if (x < 0.0) {
        return (CUDART_NAN);
    }

    if (x > 1.0) {
        if (isinf(x)) {
            return 0.0;
        }
        return ellpk(1/x)/sqrt(x);
    }

    if (x > MACHEP) {
        return (polevl<10>(x, P) - log(x) * polevl<10>(x, Q));
    }
    else {
        if (x == 0.0) {
            return (CUDART_INF);
        }
        else {
            return (C1 - 0.5 * log(x));
        }
    }
}


static __device__ double ellpkm1(double x)
{
    return ellpk(1 - x);
}

"""

ellipkm1 = _core.create_ufunc(
    'cupyx_scipy_special_ellipk',
    ('f->f', 'd->d'),
    'out0 = ellpk(in0)',
    preamble=polevl_definition+ellpk_definition,
    doc="""ellpkm1.

    Args:
        x (cupy.ndarray): The input of digamma function.

    Returns:
        cupy.ndarray: Computed value of digamma function.

    .. seealso:: :data:`scipy.special.digamma`

    """)


ellipk = _core.create_ufunc(
    'cupyx_scipy_special_ellipkm1',
    ("f->f", "d->d"),
    'out0 = ellpkm1(in0)',
    preamble=polevl_definition+ellpk_definition,
    doc="""ellpk.

    Args:
        x (cupy.ndarray): The input of digamma function.

    Returns:
        cupy.ndarray: Computed value of digamma function.

    .. seealso:: :data:`scipy.special.digamma`

    """)


ellipj_preamble = """
#include <cupy/math_constants.h>

__constant__ double M_PI_2 = 1.57079632679489661923;

static __device__ double ellipj(double u, double m, double* sn,
                                double* cn, double *dn, double *ph)
{

    double MACHEP = 1.11022302462515654042E-16;	/* 2**-53 */

    double ai, b, phi, t, twon, dnfac;
    double a[9], c[9];
    int i;

    /* Check for special cases */
    if (m < 0.0 || m > 1.0 || isnan(m)) {
        *sn = CUDART_NAN;
        *cn = CUDART_NAN;
        *ph = CUDART_NAN;
        *dn = CUDART_NAN;
        return (-1);
    }
    if (m < 1.0e-9) {
        t = sin(u);
        b = cos(u);
        ai = 0.25 * m * (u - t * b);
        *sn = t - ai * b;
        *cn = b + ai * t;
        *ph = u - ai;
        *dn = 1.0 - 0.5 * m * t * t;
        return (0);
    }
    if (m >= 0.9999999999) {
        ai = 0.25 * (1.0 - m);
        b = cosh(u);
        t = tanh(u);
        phi = 1.0 / b;
        twon = b * sinh(u);
        *sn = t + ai * (twon - u) / (b * b);
        *ph = 2.0 * atan(exp(u)) - M_PI_2 + ai * (twon - u) / b;
        ai *= t * phi;
        *cn = phi - ai * (twon - u);
        *dn = phi + ai * (twon + u);
        return (0);
    }

    /* A. G. M. scale. See DLMF 22.20(ii) */
    a[0] = 1.0;
    b = sqrt(1.0 - m);
    c[0] = sqrt(m);
    twon = 1.0;
    i = 0;

    while (fabs(c[i] / a[i]) > MACHEP) {
        if (i > 7) {
            goto done;
        }
        ai = a[i];
        ++i;
        c[i] = (ai - b) / 2.0;
        t = sqrt(ai * b);
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }

 done:
    /* backward recurrence */
    phi = twon * a[i] * u;
    do {
        t = c[i] * sin(phi) / a[i];
        b = phi;
        phi = (asin(t) + phi) / 2.0;
    }
    while (--i);

    *sn = sin(phi);
    t = cos(phi);
    *cn = t;
    dnfac = cos(phi - b);
    /* See discussion after DLMF 22.20.5 */
    if (fabs(dnfac) < 0.1) {
        *dn = sqrt(1 - m*(*sn)*(*sn));
    }
    else {
        *dn = t / dnfac;
    }
    *ph = phi;
    return (0);
}

"""


ellipj = _core.create_ufunc(
    'cupyx_scipy_special_ellipj',
    ('ff->ffff', 'dd->dddd'),
    '''
        double sn, cn, dn, ph; ellipj(in0, in1, &sn, &cn, &dn, &ph);
        out0 = sn; out1 = cn; out2 = dn; out3 = ph;
    ''',
    preamble=ellipj_preamble,
    doc="""ellipj

     Args:
         u (cupy.ndarray): The input of ellipj function.
         m (cupy.ndarray): The input of ellipj function.


     Returns:
        sn, cn, dn, ph: Computed values.

     .. seealso:: :data:`scipy.special.ellipj`
    """
)
