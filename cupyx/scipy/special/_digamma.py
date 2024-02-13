"""Digamma function

See cupy/_core/include/cupy/special/digamma.h for copyright information.

polevl below from Cephes Math Library Release 2.1: December, 1988
Copyright 1984, 1987, 1988 by Stephen L. Moshier

polevl_definition is kept because it is used elsewhere in CuPy,
although it is now no longer used in digamma.
"""

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


digamma_preamble = "#include <cupy/special/digamma.h>"


digamma = _core.create_ufunc(
    'cupyx_scipy_special_digamma', ('f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = special::digamma(in0)',
    preamble=digamma_preamble,
    doc="""The digamma function.

    Args:
        x (cupy.ndarray): The input of digamma function.

    Returns:
        cupy.ndarray: Computed value of digamma function.

    .. seealso:: :data:`scipy.special.digamma`

    """)
