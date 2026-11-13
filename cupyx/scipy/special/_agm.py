from __future__ import annotations

from cupy import _core


agm = _core.create_ufunc(
    'cupyx_scipy_special_agm', ('ff->f', 'dd->d'),
    'out0 = xsf::agm(in0, in1)',
    preamble='#include <cupy/xsf/agm.h>',
    doc='''Arithmetic-geometric mean.

    .. seealso:: :func:`scipy.special.agm`

    ''')
