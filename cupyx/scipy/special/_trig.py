"""complex-valued trig functions adapted from SciPy's cython code:

https://github.com/scipy/scipy/blob/master/scipy/special/_trig.pxd

Note: The CUDA Math library defines the real-valued cospi, sinpi
"""

csinpi_definition = """

#include <cupy/math_constants.h>

// Compute sin(pi*z) for complex arguments

__device__ complex<double> csinpi(complex<double> z)
{
    double x = z.real();
    double piy = M_PI*z.imag();
    double abspiy = abs(piy);
    double sinpix = sinpi(x);
    double cospix = cospi(x);
    double exphpiy, coshfac, sinhfac;

    if (abspiy < 700) {
        return complex<double>(sinpix*cosh(piy), cospix*sinh(piy));
    }

    /* Have to be careful--sinh/cosh could overflow while cos/sin are
     * small. At this large of values
     *
     * cosh(y) ~ exp(y)/2
     * sinh(y) ~ sgn(y)*exp(y)/2
     *
     * so we can compute exp(y/2), scale by the right factor of sin/cos
     * and then multiply by exp(y/2) to avoid overflow.
     */
    exphpiy = exp(abspiy/2.0);
    if (exphpiy == CUDART_INF) {
        if (sinpix == 0.0) {
            // Preserve the sign of zero
            coshfac = copysign(0.0, sinpix);
        } else {
            coshfac = copysign(CUDART_INF, sinpix);
        }
        if (cospix == 0.0) {
            sinhfac = copysign(0.0, cospix);
        } else {
            sinhfac = copysign(CUDART_INF, cospix);
        }
        return complex<double>(coshfac, sinhfac);
    }
    coshfac = 0.5*sinpix*exphpiy;
    sinhfac = 0.5*cospix*exphpiy;
    return complex<double>(coshfac*exphpiy, sinhfac*exphpiy);
}
"""


ccospi_definition = """

#include <cupy/math_constants.h>

// Compute cos(pi*z) for complex arguments

__device__ complex<double> csinpi(complex<double> z)
{
    double x = z.real();
    double piy = M_PI*z.imag();
    double abspiy = abs(piy);
    double sinpix = sinpi(x);
    double cospix = cospi(x);
    double exphpiy, coshfac, sinhfac;

    if (abspiy < 700) {
        return complex<double>(cospix*cosh(piy), -sinpix*sinh(piy));
    }

    // See csinpi(z) for an idea of what's going on here
    exphpiy = exp(abspiy/2.0);
    if (exphpiy == CUDART_INF) {
        if (sinpix == 0.0) {
            // Preserve the sign of zero
            coshfac = copysign(0.0, cospix);
        } else {
            coshfac = copysign(CUDART_INF, cospix);
        }
        if (cospix == 0.0) {
            sinhfac = copysign(0.0, sinpix);
        } else {
            sinhfac = copysign(CUDART_INF, sinpix);
        }
        return complex<double>(coshfac, sinhfac);
    }
    coshfac = 0.5*cospix*exphpiy;
    sinhfac = 0.5*sinpix*exphpiy;
    return complex<double>(coshfac*exphpiy, sinhfac*exphpiy);
}
"""
