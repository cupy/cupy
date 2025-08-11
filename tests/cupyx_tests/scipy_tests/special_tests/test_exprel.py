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


@testing.with_requires("scipy>=1.15")
class Testexprel:

    @testing.for_float_dtypes()
    @numpy_cupy_allclose(scipy_name="scp")
    def test_exprel(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(-1, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_2(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(1, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_large_values(self, xp, scp, dtype):
        if xp.dtype(dtype).char in 'bB':
            return xp.array(0)  # Skip to avoid overflow
        return scp.special.exprel(xp.array(720, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_small_value(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(1e-17, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_zero_values(self, xp, scp, dtype):
        return scp.special.exprel(xp.array(0, dtype=dtype))

    @testing.for_all_dtypes(no_complex=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_array_inputs(self, xp, scp, dtype):
        n = testing.shaped_arange((5, 1, 2), xp, dtype) * 0.001
        return scp.special.exprel(n)

    @testing.for_all_dtypes(no_complex=True)
    @numpy_cupy_allclose(scipy_name="scp", atol=atol, rtol=rtol)
    def test_exprel_array_inputs_2(self, xp, scp, dtype):
        n = testing.shaped_random((5, 3), xp, dtype) * 0.001
        return scp.special.exprel(n)
