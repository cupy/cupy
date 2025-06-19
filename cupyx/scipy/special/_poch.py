"""
The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/poch.c

Cephes Math Library Release 2.0:  April, 1987
Copyright 1984, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
"""

from cupy import _core

from cupyx.scipy.special._gammasgn import gammasgn_definition
from cupyx.scipy.special._gammainc import lgam_definition


poch_definition = (
    lgam_definition
    + gammasgn_definition
    + """
/*
 * Pochhammer symbol (a)_m = gamma(a + m) / gamma(a)
 */

// include for CUDART_NAN, CUDART_INF
#include <cupy/math_constants.h>


__device__ double is_nonpos_int(double x)
{
    return x <= 0 && x == ceil(x) && fabs(x) < 1e13;
}

__device__ double poch(double a, double m)
{
    double r;

    r = 1.0;

    /*
     * 1. Reduce magnitude of `m` to |m| < 1 by using recurrence relations.
     *
     * This may end up in over/underflow, but then the function itself either
     * diverges or goes to zero. In case the remainder goes to the opposite
     * direction, we end up returning 0*INF = NAN, which is OK.
     */

    /* Recurse down */
    while (m >= 1.0) {
        if (a + m == 1) {
            break;
        }
        m -= 1.0;
        r *= (a + m);
        if (!isfinite(r) || r == 0) {
            break;
        }
    }

    /* Recurse up */
    while (m <= -1.0) {
        if (a + m == 0) {
            break;
        }
        r /= (a + m);
        m += 1.0;
        if (!isfinite(r) || r == 0) {
            break;
        }
    }

    /*
     * 2. Evaluate function with reduced `m`
     *
     * Now either `m` is not big, or the `r` product has over/underflown.
     * If so, the function itself does similarly.
     */

    if (m == 0) {
        /* Easy case */
        return r;
    }
    else if (a > 1e4 && fabs(m) <= 1) {
        /* Avoid loss of precision */
        return r * pow(a, m) * (
            1
            + m*(m-1)/(2*a)
            + m*(m-1)*(m-2)*(3*m-1)/(24*a*a)
            + m*m*(m-1)*(m-1)*(m-2)*(m-3)/(48*a*a*a)
            );
    }

    /* Check for infinity */
    if (is_nonpos_int(a + m) && !is_nonpos_int(a) && a + m != m) {
        return CUDART_INF;
    }

    /* Check for zero */
    if (!is_nonpos_int(a + m) && is_nonpos_int(a)) {
        return 0;
    }

    return r * exp(lgam(a + m) - lgam(a)) * gammasgn(a + m) * gammasgn(a);
}
"""
)

poch = _core.create_ufunc(
    "cupyx_scipy_poch",
    ("ff->f", "dd->d"),
    "out0 = out0_type(poch(in0, in1));",
    preamble=poch_definition,
    doc="""Elementwise function for scipy.special.poch (Pochhammer symbol)

    .. seealso:: :meth:`scipy.special.poch`

    """,
)
