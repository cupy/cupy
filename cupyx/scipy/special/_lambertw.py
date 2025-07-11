"""Lambert W function"""
from __future__ import annotations


from cupy import _core


lambertw_preamble = "#include <cupy/xsf/lambertw.h>"

_lambertw_scalar = _core.create_ufunc(
    "cupyx_scipy_lambertw_scalar",
    ("Dld->D", "Fif->f"),
    "out0 = xsf::lambertw(in0, in1, in2)",
    preamble=lambertw_preamble,
    doc='''Internal function. Do not use.''')


def lambertw(z, k=0, tol=1e-8):
    """Lambert W function.

    .. seealso:: :meth:`scipy.special.lambertw`

    """
    return _lambertw_scalar(z, k, tol)
