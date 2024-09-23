"""Circular and hyperbolic sine and cosine integrals."""

from cupy import _core

sici = _core.create_ufunc(
    'cupyx_scipy_special_sici',
    (
        ('f->ff', 'float si, ci; xsf::sici(in0, &si, &ci); out0 = si; out1 = ci;'),
        ('d->dd', 'double si, ci; xsf::sici(in0, &si, &ci); out0 = si; out1 = ci;'),
        ('F->FF', 'complex<float> si, ci; xsf::sici(in0, &si, &ci); out0 = si; out1 = ci;'),
        ('D->DD', 'complex<double> si, ci; xsf::sici(in0, &si, &ci); out0 = si; out1 = ci;'),
    ),
    routine=None,
    preamble="#include <cupy/xsf/sici.h>",
    doc="""sici

    Sine and Cosine integrals

    .. seealso:: :meth:`scipy.special.sici`
  
    """
)
        
