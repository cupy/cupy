from __future__ import annotations

from cupy import _core


erfi = _core.create_ufunc(
    'cupyx_scipy_special_erfi', ('f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = xsf::erfi(in0)',
    preamble='#include <cupy/xsf/erf.h>',
    doc='''Imaginary error function.

    .. seealso:: :func:`scipy.special.erfi`

    ''')
