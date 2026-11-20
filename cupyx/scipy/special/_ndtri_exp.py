from __future__ import annotations

from cupy import _core


ndtri_exp = _core.create_ufunc(
    'cupyx_scipy_special_ndtri_exp', ('f->f', 'd->d'),
    'out0 = xsf::ndtri_exp(in0)',
    preamble='#include <cupy/xsf/ndtri_exp.h>',
    doc='''Inverse of :func:`log_ndtr` vs x.

    .. seealso:: :func:`scipy.special.ndtri_exp`

    ''')
