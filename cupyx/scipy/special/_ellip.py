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
    'cupyx_scipy_special_ellipkm1', ('d->d',),
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
    'cupyx_scipy_special_ellipkm1', ('d->d',),
    'out0 = ellpkm1(in0)',
    preamble=polevl_definition+ellpk_definition,
    doc="""ellpk.

    Args:
        x (cupy.ndarray): The input of digamma function.

    Returns:
        cupy.ndarray: Computed value of digamma function.

    .. seealso:: :data:`scipy.special.digamma`

    """)
