"""Assorted statistical functions"""
from __future__ import annotations

from cupy._core import ElementwiseKernel

preamble = """
#include <cupy/xsf/stats.h>
#include <cupy/xsf/cupy.h>
"""

_poisson_binom_cdf_all = ElementwiseKernel(
    in_params="T(n) p",
    out_params="T(n+1) out",
    operation=(
        "xsf::poisson_binom_cdf_all(xsf::as_mdspan(p), xsf::as_mdspan(out));"
    ),
    name="cupy_poisson_binom_cdf_all",
    preamble=preamble,
)


_take_from_discrete_cdf = ElementwiseKernel(
    in_params="T(n) cdf, int64 k",
    out_params="T out",
    operation="out = xsf::take_from_discrete_cdf(xsf::as_mdspan(cdf), k);",
    name="cupy_take_from_discrete_cdf",
    preamble=preamble,
)


def poisson_binom_cdf(k, p):
    """Poisson binomial cumulative distribution function.

    .. seealso:: :meth:`scipy.special.poisson_binom_cdf`

    """
    return _take_from_discrete_cdf(_poisson_binom_cdf_all(p), k)
