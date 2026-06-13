"""Assorted statistical functions"""
from __future__ import annotations

import numpy

import cupy
from cupy._core import BatchwiseKernel

preamble = """
#include <cupy/xsf/stats.h>
"""


_poisson_binom_pmf_all = BatchwiseKernel(
    in_params="T(n) p",
    out_params="T(n+1) out",
    operation=(
        "xsf::poisson_binom_pmf_all(p.as_mdspan(), out.as_mdspan());"
    ),
    name="cupy_poisson_binom_pmf_all",
    preamble=preamble,
)


_take_from_pmf = BatchwiseKernel(
    in_params="T(n) pmf, int64 k",
    out_params="T out",
    operation="out = xsf::take_from_pmf(pmf.as_mdspan(), k);",
    name="cupy_take_from_pmf",
    preamble=preamble,
)


_poisson_binom_cdf_all = BatchwiseKernel(
    in_params="T(n) p",
    out_params="T(n+1) out",
    operation=(
        "xsf::poisson_binom_cdf_all(p.as_mdspan(), out.as_mdspan());"
    ),
    name="cupy_poisson_binom_cdf_all",
    preamble=preamble,
)


_take_from_discrete_cdf = BatchwiseKernel(
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
    p, k = cupy.asarray(p), cupy.asarray(k)
    batch_shape = p.shape[:-1]
    n = p.shape[-1]
    intermediate_pmf = cupy.empty(batch_shape + (n + 1,), dtype=p.dtype)
    _poisson_binom_pmf_all(p, out=intermediate_pmf)
    out = cupy.empty(numpy.broadcast_shapes(
        batch_shape, k.shape), dtype=p.dtype)
    _take_from_pmf(intermediate_pmf, k, out=out)
    return out


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
    p, k = cupy.asarray(p), cupy.asarray(k)
    batch_shape = p.shape[:-1]
    n = p.shape[-1]
    intermediate_cdf = cupy.empty(batch_shape + (n + 1,), dtype=p.dtype)
    _poisson_binom_cdf_all(p, out=intermediate_cdf)
    out = cupy.empty(numpy.broadcast_shapes(
        batch_shape, k.shape), dtype=p.dtype)
    _take_from_discrete_cdf(intermediate_cdf, k, out=out)
    return out
