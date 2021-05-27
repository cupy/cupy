# This source code contains SciPy's code.
# https://github.com/scipy/scipy/blob/master/scipy/special/cephes/zeta.c
#
#
# Cephes Math Library Release 2.0:  April, 1987
# Copyright 1984, 1987 by Stephen L. Moshier
# Direct inquiries to 30 Frost Street, Cambridge, MA 02140

# TODO(YoshikawaMasashi): float implementation of zeta function

from cupy import _core


zeta_definition = '''
/* Expansion coefficients
 * for Euler-Maclaurin summation formula
 * (2k)! / B2k
 * where B2k are Bernoulli numbers
 */
__constant__ double A[] = {
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9,	/*1.307674368e12/691 */
    7.47242496e10,
    -2.950130727918164224e12,	/*1.067062284288e16/3617 */
    1.1646782814350067249e14,	/*5.109094217170944e18/43867 */
    -4.5979787224074726105e15,	/*8.028576626982912e20/174611 */
    1.8152105401943546773e17,	/*1.5511210043330985984e23/854513 */
    -7.1661652561756670113e18	/*1.6938241367317436694528e27/236364091 */
};

__constant__ double MACHEP = 1.11022302462515654042E-16;

/* 30 Nov 86 -- error in third coefficient fixed */


double __device__ zeta(double x, double q)
{
    int i;
    double a, b, k, s, t, w;

    if (x == 1.0){
        return 1.0/0.0;
    }

    if (x < 1.0) {
        return nan("");
    }

    if (q <= 0.0) {
        if (q == floor(q)) {
            return 1.0/0.0;
        }
        if (x != floor(x)){
            return nan("");	/* because q^-x not defined */
        }
    }

    /* Asymptotic expansion
     * http://dlmf.nist.gov/25.11#E43
     */
    if (q > 1e8) {
        return (1/(x - 1) + 1/(2*q)) * pow(q, 1 - x);
    }

    /* Euler-Maclaurin summation formula */

    /* Permit negative q but continue sum until n+q > +9 .
     * This case should be handled by a reflection formula.
     * If q<0 and x is an integer, there is a relation to
     * the polyGamma function.
     */
    s = pow(q, -x);
    a = q;
    i = 0;
    b = 0.0;
    while ((i < 9) || (a <= 9.0)) {
        i += 1;
        a += 1.0;
        b = pow(a, -x);
        s += b;
        if (fabs(b / s) < MACHEP){
            return s;
        }
    }

    w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    a = 1.0;
    k = 0.0;
    for (i = 0; i < 12; i++) {
        a *= x + k;
        b /= w;
        t = a * b / A[i];
        s = s + t;
        t = fabs(t / s);
        if (t < MACHEP){
            return s;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
    return s;
}
'''


zeta = _core.create_ufunc(
    'cupyx_scipy_zeta', ('ff->f', 'dd->d'),
    'out0 = zeta(in0, in1)',
    preamble=zeta_definition,
    doc="""Hurwitz zeta function.

    Args:
        x (cupy.ndarray): Input data, must be real.
        q (cupy.ndarray): Input data, must be real.

    Returns:
        cupy.ndarray: Values of zeta(x, q).

    .. seealso:: :data:`scipy.special.zeta`

    """)
