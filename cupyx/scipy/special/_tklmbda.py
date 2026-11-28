from __future__ import annotations

from cupy import _core


tklmbda = _core.create_ufunc(
    'cupyx_scipy_special_tklmbda', ('ff->f', 'dd->d'),
    'out0 = xsf::tukeylambdacdf(in0, in1)',
    preamble='#include <cupy/xsf/stats.h>',
    doc='''Cumulative distribution function of the Tukey lambda distribution.

    .. seealso:: :func:`scipy.special.tklmbda`

    ''')
