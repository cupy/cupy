from __future__ import annotations

from cupy import _core


wofz = _core.create_ufunc(
    'cupyx_scipy_special_wofz', ('F->F', 'D->D'),
    'out0 = xsf::wofz(in0)',
    preamble='#include <cupy/xsf/erf.h>',
    doc='''Faddeeva function.

    .. seealso:: :meth:`scipy.special.wofz`

    ''')
