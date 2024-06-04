from cupy import _core


wright_bessel_preamble = "#include <cupy/special/wright_bessel.h>"

wright_bessel = _core.create_ufunc(
    'cupyx_scipy_special_wright_bessel', ('fff->f', 'ddd->d'),
    'out0 = special::wright_bessel(in0, in1, in2)',
    preamble=wright_bessel_preamble,
    doc="""Wright's generalized Bessel function

    .. seealso:: :meth:`scipy.special.wright_bessel`

    """)
