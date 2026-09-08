from __future__ import annotations

from cupy import _core


voigt_profile = _core.create_ufunc(
    'cupyx_scipy_special_voigt_profile', ('fff->f', 'ddd->d'),
    'out0 = xsf::voigt_profile(in0, in1, in2)',
    preamble='#include <cupy/xsf/erf.h>',
    doc='''Voigt profile.

    .. seealso:: :func:`scipy.special.voigt_profile`

    ''')
