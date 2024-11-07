from cupy import testing


@testing.with_requires('scipy')
class TestLambertW:

    @testing.for_dtypes('fd')
    @testing.for_dtypes('il', name='branch_dtype')
    @testing.numpy_cupy_allclose(atol=1e-13, rtol=1e-10, scipy_name='scp')
    def test_values(self, xp, scp, dtype, branch_dtype):
        k = xp.repeat(xp.arange(-3, 3, dtype=branch_dtype), 24)
        x = xp.linspace(-20, 20, 12, dtype=dtype)
        y = xp.linspace(-20, 20, 12, dtype=dtype)
        x, y = xp.meshgrid(x, y)
        z = (x + y*1j).ravel()

        return scp.special.lambertw(z, k)
