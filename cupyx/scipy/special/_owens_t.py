from __future__ import annotations

from cupy import _core


owens_t = _core.create_ufunc(
    'cupyx_scipy_special_owens_t', ('ff->f', 'dd->d'),
    'out0 = xsf::owens_t(in0, in1)',
    preamble='#include <cupy/xsf/stats.h>',
    doc='''Owen's T Function.

    .. seealso:: :func:`scipy.special.owens_t`

    ''')
