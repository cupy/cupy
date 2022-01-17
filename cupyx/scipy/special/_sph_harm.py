"""
The source code here is an adaptation with minimal changes from the following
SciPy Cython file:

https://github.com/scipy/scipy/blob/master/scipy/special/sph_harm.pxd
"""

from cupy import _core

from cupyx.scipy.special._poch import poch_definition
from cupyx.scipy.special._lpmv import lpmv_definition

sph_harmonic_definition = (
    poch_definition
    + lpmv_definition
    + """

#include <cupy/complex.cuh>

// include for CUDART_NAN, CUDART_INF
#include <cupy/math_constants.h>

#define NPY_PI        3.141592653589793238462643383279502884  /* pi */

// from scipy/special/sph_harm.pxd
__device__ complex<double> sph_harmonic(int m, int n, double theta, double phi)
{
    double x, prefactor;
    complex<double> val;
    int mp;

    x = cos(phi);
    if (abs(m) > n)
    {
        // sf_error.error("sph_harm", sf_error.ARG,
        //                "m should not be greater than n")
        return CUDART_NAN;
    }
    if (n < 0)
    {
        // sf_error.error("sph_harm", sf_error.ARG, "n should not be negative")
        return CUDART_NAN;
    }
    if (m < 0)
    {
        mp = -m;
        prefactor = poch(n + mp + 1, -2 * mp);
        if ((mp % 2) == 1)
        {
            prefactor = -prefactor;
        }
    }
    else
    {
        mp = m;
    }
    val = pmv_wrap(mp, n, x);
    if (m < 0)
    {
        val *= prefactor;
    }
    val *= sqrt((2*n + 1) / 4.0 / NPY_PI);
    val *= sqrt(poch(n + m + 1, -2 * m));

    complex<double> exponent(0, m * theta);
    val *= exp(exponent);

    return val;
}
"""
)


sph_harm = _core.create_ufunc(
    "cupyx_scipy_lpmv",
    ("iiff->F", "iidd->D", "llff->F", "lldd->D"),
    "out0 = out0_type(sph_harmonic(in0, in1, in2, in3));",
    preamble=sph_harmonic_definition,
    doc="""Spherical Harmonic.

    .. seealso:: :meth:`scipy.special.sph_harm`

    """,
)
