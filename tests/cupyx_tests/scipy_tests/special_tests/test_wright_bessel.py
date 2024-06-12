from cupy import testing


@testing.with_requires('scipy')
class TestWrightBessel:
    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_values(self, xp, scp, dtype):
        a = 2**xp.arange(-3, 4, dtype=dtype)
        b = 2**xp.arange(-3, 4, dtype=dtype)
        x = 2**xp.arange(-3, 4, dtype=dtype)
        a, b, x = xp.meshgrid(a, b, x)
        a, b, x = a.ravel(), b.ravel(), x.ravel()

        return scp.special.wright_bessel(a, b, x)
