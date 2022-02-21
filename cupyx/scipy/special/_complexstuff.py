"""complex-valued functions adapted from SciPy's cython code:

https://github.com/scipy/scipy/blob/master/scipy/special/_complexstuff.pxd

Notes:
- instead of zabs, use thrust::abs
- instead of zarg, use thrust::arg
- instead of zcos, use thrust::cos
- instead of zexp, use thrust::exp
- instead of zisfinite, use isfinite defined in _core/include/cupy/complex.cuh
- instead of zisinf, use isinf defined in _core/include/cupy/complex.cuh
- instead of zisnan, use isnan defined in _core/include/cupy/complex.cuh
- instead of zpack, use complex<double>(real, imag)
- instead of zpow, use thrust::pow
- instead of zreal, use z.real()
- instead of zsin, use thrust::sin
- instead of zsqrt, use thrust::sqrt

"""


zlog1_definition = """

/* Compute log, paying special attention to accuracy around 1. We
 * implement this ourselves because some systems (most notably the
 * Travis CI machines) are weak in this regime.
 */

#define TOL_ZLOG1 2.220446092504131e-16


__device__ complex<double> zlog1(complex<double> z)
{
    complex<double> coeff = -1.0;
    complex<double> res = 0.0;

    if (abs(z - 1.0) > 0.1) {
        return log(z);  // complex log via Thrust
    }
    z = z - 1.0;
    if (z == 0.0) {
        return 0;
    }
    for (int n=1; n<17; n++)
    {
        coeff *= -z;
        res += coeff / complex<double>(n, 0);
        if (abs(res/coeff) < TOL_ZLOG1) {
            break;
        }
    }
    return res;
}
"""
