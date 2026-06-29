"""Assorted statistical functions"""
from __future__ import annotations

from cupy._core import ElementwiseKernel

preamble = """
#include <cupy/xsf/stats.h>
"""


_poisson_binom_pmf_all = ElementwiseKernel(
    in_params="T(n) p",
    out_params="T(n+1) out",
    operation=(
        "xsf::poisson_binom_pmf_all(p.as_mdspan(), out.as_mdspan());"
    ),
    name="cupy_poisson_binom_pmf_all",
    preamble=preamble,
)


_take_from_pmf = ElementwiseKernel(
    in_params="T(n) pmf, int64 k",
    out_params="T out",
    operation="out = xsf::take_from_pmf(pmf.as_mdspan(), k);",
    name="cupy_take_from_pmf",
    preamble=preamble,
)


_poisson_binom_cdf_all = ElementwiseKernel(
    in_params="T(n) p",
    out_params="T(n+1) out",
    operation=(
        "xsf::poisson_binom_cdf_all(p.as_mdspan(), out.as_mdspan());"
    ),
    name="cupy_poisson_binom_cdf_all",
    preamble=preamble,
)


_take_from_discrete_cdf = ElementwiseKernel(
    in_params="T(n) cdf, int64 k",
    out_params="T out",
    operation="out = xsf::take_from_discrete_cdf(cdf.as_mdspan(), k);",
    name="cupy_take_from_discrete_cdf",
    preamble=preamble,
)


def _poisson_binom_pmf(k, p):
    """Returns pmf of Poisson Binomial distribution.

    Parameters
    ----------
    k : array
        Number of successes at which to evaluate pmf.

    p : array
        Success probabilities of independent Bernoulli trials.

    Notes
    -----
    This is equivalent to a gufunc with signature ``()(i)->()``.
    The last dimension of `p` contains success probabilities and
    the preceding dimensions are batch dimensions. The batch
    dimensions are broadcast against ``k``.
    """
    return _take_from_pmf(_poisson_binom_pmf_all(p), k)


def _poisson_binom_cdf(k, p):
    """Returns cdf of Poisson Binomial distribution.

    Parameters
    ----------
    k : array
        Number of successes at which to evaluate cdf.

    p : array
        Success probabilities of independent Bernoulli trials.

    Notes
    -----
    This is equivalent to a gufunc with signature ``()(i)->()``.
    The last dimension of `p` contains success probabilities and
    the preceding dimensions are batch dimensions. The batch
    dimensions are broadcast against ``k``.
    """
    return _take_from_discrete_cdf(_poisson_binom_cdf_all(p), k)
