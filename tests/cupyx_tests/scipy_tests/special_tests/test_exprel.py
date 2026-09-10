import cupyx.scipy.special  # NOQA

import cupy

from cupy import testing
from cupy.testing import numpy_cupy_allclose

try:
    import scipy.special  # NOQA
except ImportError:
    pass


atol = {
    'default': 1e-6,
    cupy.float16: 1e-2,
}
rtol = {
    'default': 1e-6,
    cupy.float16: 1e-2,
}


# NOTE: float16 is left out of the dtype sweeps below.  SciPy 1.18 resolves
# float16 input to the float32 loop of the `scipy.special` ufuncs and returns
# float32, whereas CuPy's ufuncs declare `e->d` and return float64.  The
# float16 tolerances above are kept for when that coverage comes back.
# TODO: switch those loops to `e->f` and restore float16 coverage before the
# minimum SciPy version is 1.18.
@testing.with_requires("scipy>=1.15")
class Testexprel:

    @testing.for_float_dtypes(no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp")
    def test_exprel(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(-1, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_2(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(1, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_large_values(self, xp, scp, dtype):
        if xp.dtype(dtype).char in 'bB':
            return xp.array(0)  # Skip to avoid overflow
        return scp.special.exprel(xp.array(720, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_small_value(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(1e-17, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_zero_values(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(0, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_array_inputs(self, xp, scp, dtype):
        n = testing.shaped_arange((5, 1, 2), xp, dtype) * 0.001
        return scp.special.exprel(n)

    @testing.for_all_dtypes(no_complex=True, no_float16=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_array_inputs_2(self, xp, scp, dtype):
        n = testing.shaped_random((5, 3), xp, dtype) * 0.001
        return scp.special.exprel(n)
