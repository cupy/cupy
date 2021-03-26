# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/master/scipy/special/cephes/psi.c
#
#
# Cephes Math Library Release 2.8:  June, 2000
# Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
#
#
# Code for the rational approximation on [1, 2] is:
#
# (C) Copyright John Maddock 2006.
# Use, modification and distribution are subject to the
# Boost Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from cupy import _core


polevl_definition = '''
template<int N> static __device__ double polevl(double x, double coef[])
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


psi_definition = '''
__constant__ double A[] = {
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2
};

__constant__ double PI = 3.141592653589793;
__constant__ double EULER = 0.5772156649015329;

__constant__ float Y = 0.99558162689208984f;

__constant__ double root1 = 1569415565.0 / 1073741824.0;
__constant__ double root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
__constant__ double root3 = 0.9016312093258695918615325266959189453125e-19;

__constant__ double P[] = {
    -0.0020713321167745952,
    -0.045251321448739056,
    -0.28919126444774784,
    -0.65031853770896507,
    -0.32555031186804491,
    0.25479851061131551
};
__constant__ double Q[] = {
    -0.55789841321675513e-6,
    0.0021284987017821144,
    0.054151797245674225,
    0.43593529692665969,
    1.4606242909763515,
    2.0767117023730469,
    1.0
};

static __device__ double digamma_imp_1_2(double x)
{
    /*
     * Rational approximation on [1, 2] taken from Boost.
     *
     * Now for the approximation, we use the form:
     *
     * digamma(x) = (x - root) * (Y + R(x-1))
     *
     * Where root is the location of the positive root of digamma,
     * Y is a constant, and R is optimised for low absolute error
     * compared to Y.
     *
     * Maximum Deviation Found:               1.466e-18
     * At double precision, max error found:  2.452e-17
     */
    double r, g;
    g = x - root1 - root2 - root3;
    r = polevl<5>(x - 1.0, P) / polevl<6>(x - 1.0, Q);

    return g * Y + g * r;
}


static __device__ double psi_asy(double x)
{
    double y, z;

    if (x < 1.0e17) {
        z = 1.0 / (x * x);
        y = z * polevl<6>(z, A);
    }
    else {
        y = 0.0;
    }

    return log(x) - (0.5 / x) - y;
}


double __device__ psi(double x)
{
    double y = 0.0;
    double q, r;
    int i, n;

    if (isnan(x)) {
        return x;
    }
    else if (isinf(x)){
        if(x > 0){
            return x;
        }else{
            return nan("");
        }
    }
    else if (x == 0) {
        return -1.0/0.0;
    }
    else if (x < 0.0) {
        /* argument reduction before evaluating tan(pi * x) */
        r = modf(x, &q);
        if (r == 0.0) {
            return nan("");
        }
        y = -PI / tan(PI * r);
        x = 1.0 - x;
    }

    /* check for positive integer up to 10 */
    if ((x <= 10.0) && (x == floor(x))) {
        n = (int)x;
        for (i = 1; i < n; i++) {
            y += 1.0 / i;
        }
        y -= EULER;
        return y;
    }

    /* use the recurrence relation to move x into [1, 2] */
    if (x < 1.0) {
        y -= 1.0 / x;
        x += 1.0;
    }
    else if (x < 10.0) {
        while (x > 2.0) {
            x -= 1.0;
            y += 1.0 / x;
        }
    }
    if ((1.0 <= x) && (x <= 2.0)) {
        y += digamma_imp_1_2(x);
        return y;
    }

    /* x is large, use the asymptotic series */
    y += psi_asy(x);
    return y;
}
'''


digamma = _core.create_ufunc(
    'cupyx_scipy_digamma', ('f->f', 'd->d'),
    'out0 = psi(in0)',
    preamble=polevl_definition+psi_definition,
    doc="""The digamma function.

    Args:
        x (cupy.ndarray): The input of digamma function.

    Returns:
        cupy.ndarray: Computed value of digamma function.

    .. seealso:: :data:`scipy.special.digamma`

    """)
