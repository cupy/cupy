"""Circular and hyperbolic sine and cosine integrals."""
from __future__ import annotations


from cupy import _core

sici = _core.create_ufunc(
    'cupyx_scipy_special_sici',
    (
        (
            'f->ff',
            '''
            float si, ci;
            xsf::sici(in0, si, ci);
            out0 = si; out1 = ci;
            ''',
        ),
        (
            'd->dd',
            '''
            double si, ci;
            xsf::sici(in0, si, ci);
            out0 = si; out1 = ci;
            ''',
        ),
        (
            'F->FF',
            '''
            complex<float> si, ci;
            xsf::sici(in0, si, ci);
            out0 = si; out1 = ci;
            ''',
        ),
        (
            'D->DD',
            '''
            complex<double> si, ci;
            xsf::sici(in0, si, ci);
            out0 = si; out1 = ci;
            ''',
        ),
    ),
    preamble="#include <cupy/xsf/sici.h>",
    doc="""sici

    Sine and Cosine integrals

    .. seealso:: :meth:`scipy.special.sici`

    """
)

shichi = _core.create_ufunc(
    'cupyx_scipy_special_shichi',
    (
        (
            'f->ff',
            '''
            float shi, chi;
            xsf::shichi(in0, shi, chi);
            out0 = shi; out1 = chi;
            ''',
        ),
        (
            'd->dd',
            '''
            double shi, chi;
            xsf::shichi(in0, shi, chi);
            out0 = shi; out1 = chi;
            ''',
        ),
        (
            'F->FF',
            '''
            complex<float> shi, chi;
            xsf::shichi(in0, shi, chi);
            out0 = shi; out1 = chi;
            ''',
        ),
        (
            'D->DD',
            '''
            complex<double> shi, chi;
            xsf::shichi(in0, shi, chi);
            out0 = shi; out1 = chi;
            ''',
        ),
    ),
    preamble="#include <cupy/xsf/sici.h>",
    doc="""shichi

    Hyperbolic sine and cosine integrals.

    .. seealso:: :meth:`scipy.special.shichi`

    """
)
