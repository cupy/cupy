"""
The source code here is an adaptation with minimal changes from the following
files in SciPy:

https://github.com/scipy/scipy/blob/master/scipy/special/specfun_wrappers.c

Code for psi_spec, gamma2, lpmv0, lpmv was manually translated to C++ from
SciPy's Fortran-77 code located in:
https://github.com/scipy/scipy/blob/master/scipy/special/specfun/specfun.f

The fortran code in scipy originated in the following book.

    "Computation of Special Functions", 1996, John Wiley & Sons, Inc.

    Shanjie Zhang and Jianming Jin

    Copyrighted but permission granted to use code in programs.
"""

from cupy import _core

# May want to use CUDART_PI instead of redefining here.
# The two seem to differ in the last didit, though.
#     #define CUDART_PI               3.1415926535897931e+0

nan_inf = """
// include for CUDART_NAN, CUDART_INF
#include <cupy/math_constants.h>

"""


lpmv_definition = """

// include for CUDART_NAN, CUDART_INF
#include <cupy/math_constants.h>

// CUDA C++ translation of code from:
// https://github.com/scipy/scipy/blob/master/scipy/special/specfun/specfun.f

/*
 *  Purpose: Compute Psi function
 *   Input :  x  --- Argument of psi(x)
 */
__device__ double psi_spec(double x)
{
    double xa = fabs(x);
    double PI = 3.1415926535897930;
    double EL = 0.5772156649015329;  // Euler-Mascheroni constant
    double s = 0.;
    double ps;
    int n = 0;
    if ((x - (int)x == 0.) && (x <= 0.))
    {
        return CUDART_INF;
    } else if (xa - (int)xa == 0.) {
        n = (int) xa;
        for (int k=1; k <= n - 1; k++)
        {
           s += 1.0/k;
        }
        ps = -EL + s;
    } else if ((xa - 0.5) - (int)(xa - 0.5) == 0.) {
        n = (int)(xa - 0.5);
        for (int k=1; k <= n; k++)
        {
           s += 1.0/(2.0*k - 1.0);
        }
        ps = -EL + 2.0 * s - 1.386294361119891;
    } else {
        if (xa < 10.0)
        {
            n = 10 - (int)(xa);
            for (int k=1; k < n; k++)
            {
               s += 1.0/(xa + k);
            }
            xa += (double)n;
        }
        double x2 = 1.0 / (xa*xa);
        double a1 = -.8333333333333e-01;
        double a2 = .83333333333333333e-02;
        double a3 = -.39682539682539683e-02;
        double a4 = .41666666666666667e-02;
        double a5 = -.75757575757575758e-02;
        double a6 = .21092796092796093e-01;
        double a7 = -.83333333333333333e-01;
        double a8 = .44325980392156860;
        ps = log(xa) - 0.5/xa + x2*(((((((a8*x2 + a7)*x2 +
             a6)*x2 + a5)*x2 + a4)*x2 + a3)*x2 + a2)*x2 + a1);
        ps -= s;
    }
    if (x < 0.) {
        ps = ps - PI*cos(PI*x)/sin(PI*x) - 1.0/x;
    }
    return ps;
}


/*
 *       Purpose: Compute gamma function Gamma(x)
 *       Input :  x  --- Argument of Gamma(x)
 *                      ( x is not equal to 0,-1,-2,...)
 *       Output:  GA --- Gamma(x)
 */
__device__ double gamma2(double x)
{
    double ga;
    double G[26] = {1.0, 0.5772156649015329,
                   -0.6558780715202538, -0.420026350340952e-1,
                    0.1665386113822915,-.421977345555443e-1,
                    -.96219715278770e-2, .72189432466630e-2,
                    -.11651675918591e-2, -.2152416741149e-3,
                     .1280502823882e-3, -.201348547807e-4,
                    -.12504934821e-5, .11330272320e-5,
                    -.2056338417e-6, .61160950e-8,
                     .50020075e-8, -.11812746e-8,
                     .1043427e-9, .77823e-11,
                    -.36968e-11, .51e-12,
                    -.206e-13, -.54e-14, .14e-14, .1e-15};
    double PI = 3.141592653589793;
    int k;
    if ((x - (int)x) == 0.)
    {
        if (x > 0.)
        {
            ga = 1.0;
            for (k=2; k < x; k++)
            {
                ga *= k;
            }
        } else
        {
            ga = CUDART_INF;
        }
    }
    else
    {
        double r = 1.0;
        double z = fabs(x);
        if (z > 1.0){
            int m = (int)z;
            for (k=1; k<=m; k++)
            {
                r *= (z - k);
            }
            z -= m;
        } else {
            z = x;
        }
        double gr = G[24];
        for (k=25; k>=1; k--)
        {
            gr = gr*z + G[k];
        }
        ga = 1.0 / (gr * z);
        if (fabs(x) > 1.0)
        {
            ga *= r;
            if (x < 0.0)
            {
                ga = -PI / (x*ga*sin(PI*x));
            }
        }
    }
    return ga;
}


/*
 *     Purpose: Compute the associated Legendre function
 *              Pmv(x) with an integer order and an arbitrary
 *              nonnegative degree v
 *     Input :  x   --- Argument of Pm(x)  ( -1 <= x <= 1 )
 *              m   --- Order of Pmv(x)
 *              v   --- Degree of Pmv(x)
 */
__device__ double lpmv0(double v, double m, double x)
{
    double pmv, r;
    int j, k;

    double PI = 3.141592653589793;
    double EL = .5772156649015329;  // Euler-Mascheroni constant
    double EPS = 1.0e-14;
    int nv = (int)v;
    double v0 = v - nv;
    if ((x == -1.0) && (v != nv))
    {
        if (m == 0)
            return -CUDART_INF;
        else
            return CUDART_INF;
    }
    double c0 = 1.0;
    if (m != 0)
    {
        double rg = v*(v + m);
        for (j=1; j < m; j++)
        {
            rg *= (v*v - j*j);
        }
        double xq = sqrt(1.0 - x*x);

        double r0 = 1.0;
        for (j=1; j <= m; j++)
        {
            r0 *= 0.5*xq/j;
        }
        c0 = r0*rg;
    }
    if (v0 == 0.)
    {
        // DLMF 14.3.4, 14.7.17, 15.2.4
        pmv = 1.0;
        r = 1.0;
        for (k=1; k <= nv - m; k++)
        {
            r *= 0.5*(-nv + m + k - 1.0)*(nv + m + k)/(k*(k + m))*(1.0 + x);
            pmv += r;
        }
        pmv *= c0;
        if ((nv % 2) == 1)
        {
            pmv = -pmv;
        }
    }
    else
    {
        if (x >= -0.35)
        {
            // DLMF 14.3.4, 15.2.1
            pmv = 1.0;
            r = 1.0;
            for (k = 1; k <= 100; k++)
            {
                 r *= 0.5*(-v + m + k - 1.0)*(v + m + k)/(k*(m + k))*(1.0 - x);
                 pmv += r;
                 if ((k > 12) & (fabs(r/pmv) < EPS)){
                    break;
                 }

            }
            pmv *= c0;
            if (((int)m % 2) == 1)
            {
                pmv = -pmv;
            }
        }
        else
        {
            // DLMF 14.3.5, 15.8.10
            double vs = sin(v * PI) / PI;
            double pv0 = 0.0;
            double r2;
            if (m != 0)
            {
                double qr = sqrt((1.0 - x)/(1.0 + x));
                r2 = 1.0;
                for (j = 1; j <= m; j++)
                {
                    r2 *= qr*j;
                }
                double s0 = 1.0;
                double r1 = 1.0;
                for (k = 1; k < m; k++)
                {
                    r1=0.5*r1*(-v + k - 1)*(v + k)/(k*(k - m))*(1.0 + x);
                    s0 += r1;
                }
                pv0 = -vs*r2/m*s0;
            }
            double psv = psi_spec(v);
            double pa = 2.0*(psv + EL) + PI/tan(PI*v) + 1.0/v;
            double s1 = 0.0;
            for (j = 1; j <= m; j++)
            {
                s1 += (j*j + v*v)/(j*(j*j - v*v));
            }
            pmv = pa + s1 - 1.0/(m - v) + log(0.5*(1.0 + x));
            r = 1.0;
            for (k = 1; j <= 100; k++)
            {
                r *= 0.5*(-v + m + k-1.0)*(v + m + k)/(k*(k + m))*(1.0 + x);
                double s = 0.0;
                for (j = 1; j <= m; j++)
                {
                    double kjsq = (k + j) * (k + j);
                    s += (kjsq + v*v)/((k + j)*(kjsq - v*v));
                }
                double s2 = 0.0;
                for (j = 1; j <= k; j++)
                {
                    s2 += 1.0/(j*(j*j - v*v));
                }
                double pss = pa + s + 2.0*v*v*s2 - 1.0/(m + k - v)
                             + log(0.5*(1.0 + x));
                r2 = pss*r;
                pmv += r2;
                if (fabs(r2/pmv) < EPS)
                {
                    break;
                }
            }
            pmv = pv0 + pmv*vs*c0;
        }
    }
    return pmv;
}


/*       Purpose: Compute the associated Legendre function
 *                Pmv(x) with an integer order and an arbitrary
 *                degree v, using recursion for large degrees
 *       Input :  x   --- Argument of Pm(x)  ( -1 <= x <= 1 )
 *                m   --- Order of Pmv(x)
 *                v   --- Degree of Pmv(x)
 *       Output:  PMV --- Pmv(x)
 */
__device__ double lpmv(double v, int m, double x)
{
    double pmv;
    int j;

    if ((x == -1.0) && (v != (int)v))
    {
        if (m == 0)
        {
            return -CUDART_INF;
        } else
        {
            return CUDART_INF;
        }
    }

    double vx = v;
    double mx = m;
    // DLMF 14.9.5
    if (v < 0)
    {
        vx = -vx - 1;
    }
    int neg_m = 0;
    if (m < 0)
    {
        if (((vx + m + 1) > 0) || (vx != (int)vx))
        {
            neg_m = 1;
            mx = -m;
        }
        else
        {
            // We don't handle cases where DLMF 14.9.3 doesn't help
            return CUDART_NAN;
        }
    }

    int nv = (int)vx;
    double v0 = vx - nv;
    if ((nv > 2) && (nv > mx))
    {
        // Up-recursion on degree, AMS 8.5.3 / DLMF 14.10.3
        double p0 = lpmv0(v0 + mx, mx, x);
        double p1 = lpmv0(v0 + mx + 1, mx, x);
        pmv = p1;
        for (j = mx + 2; j <= nv; j++)
        {
          pmv = ((2*(v0 + j) - 1)*x*p1 - (v0 + j - 1 + mx)*p0) / (v0 + j - mx);
          p0 = p1;
          p1 = pmv;
        }
    }
    else
    {
        pmv = lpmv0(vx, mx, x);
    }

    if ((neg_m != 0) && (fabs(pmv) < 1.0e300))
    {
        double g1 = gamma2(vx - mx + 1);
        double g2 = gamma2(vx + mx + 1);
        pmv = pmv * g1/g2 * pow(-1, mx);
    }
    return pmv;
}


// pmv_wrap as in
// https://github.com/scipy/scipy/blob/master/scipy/special/specfun_wrappers.c

__device__  double pmv_wrap(double m, double v, double x){
  int int_m;
  double out;

  if (m != floor(m))
  {
      return CUDART_NAN;
  }
  int_m = (int) m;
  out = lpmv(v, int_m, x);
  // should raise an overflow warning here on INF
  return out;
}

"""

lpmv = _core.create_ufunc(
    "cupyx_scipy_lpmv",
    ("fff->f", "ddd->d"),
    "out0 = out0_type(pmv_wrap(in0, in1, in2));",
    preamble=lpmv_definition,
    doc="""Associated Legendre function of integer order and real degree.

    .. seealso:: :meth:`scipy.special.lpmv`

    """,
)
