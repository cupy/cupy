from __future__ import annotations

from cupy import _core


dawsn = _core.create_ufunc(
    'cupyx_scipy_special_dawsn', ('f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = xsf::dawsn(in0)',
    preamble='#include <cupy/xsf/erf.h>',
    doc='''Dawson's integral.

    .. seealso:: :func:`scipy.special.dawsn`

    ''')
