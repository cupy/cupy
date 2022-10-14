"""
The source code here is an adaptation with minimal changes from the following
file in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/gammasgn.c

Cephes Math Library Release 2.0:  April, 1987
Copyright 1984, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
"""

from cupy import _core


gammasgn_definition = """
__device__ double gammasgn(double x)
{
    double fx;

    if (isnan(x)) {
      return x;
    }
    if (x > 0) {
        return 1.0;
    }
    else {
        fx = floor(x);
        if (x - fx == 0.0) {
            return 0.0;
        }
        else if ((int)fx % 2) {
            return -1.0;
        }
        else {
            return 1.0;
        }
    }
}
"""

gammasgn = _core.create_ufunc(
    "cupyx_scipy_gammasgn",
    ("f->f", "d->d"),
    "out0 = out0_type(gammasgn(in0));",
    preamble=gammasgn_definition,
    doc="""Elementwise function for scipy.special.gammasgn

    .. seealso:: :meth:`scipy.special.gammasgn`

    """,
)
