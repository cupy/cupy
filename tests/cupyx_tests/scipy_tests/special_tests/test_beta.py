import numpy

from cupy import testing
import cupyx.scipy.special  # NOQA


@testing.gpu
@testing.with_requires('scipy')
class TestBeta:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_beta_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((1, 10), xp, dtype) + dtype(1)
        b = testing.shaped_arange((10, 1), xp, dtype) + dtype(1)
        return scp.special.beta(a, b)

    @testing.for_float_dtypes()  # no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_beta_linspace(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.linspace(-20, 20, 100, dtype=dtype).reshape(1, 100)
        b = xp.linspace(-20, 20, 100, dtype=dtype).reshape(100, 1)
        return scp.special.beta(a, b)

    # TODO: fix failure if smaller starting values are used for the range
    @testing.for_float_dtypes()  # no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_beta_linspace_large(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.linspace(10, 200, 200, dtype=dtype).reshape(1, 200)
        b = xp.linspace(10, 200, 200, dtype=dtype).reshape(200, 1)
        return scp.special.beta(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_beta_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        return scp.special.beta(dtype(2.5), dtype(3.5))

    def test_beta_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special
        assert special.beta(1, 1) == 1.0
        testing.assert_allclose(special.beta(-100.3, 1e-200),
                                special.gamma(1e-200))
        testing.assert_allclose(special.beta(0.0342, 171),
                                24.070498359873497, rtol=1e-13, atol=0)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_beta_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.array([-numpy.inf, numpy.nan, numpy.inf, 0], dtype=dtype)
        return scp.special.beta(a.reshape((1, 4)), a.reshape((4, 1)))


@testing.gpu
@testing.with_requires('scipy')
class TestBetaLn:

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_betaln_arange(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((1, 10), xp, dtype) + dtype(1)
        b = testing.shaped_arange((10, 1), xp, dtype) + dtype(1)
        return scp.special.betaln(a, b)

    # TODO: fix failure if smaller starting values are used for the range
    @testing.for_float_dtypes()  # no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_betaln_linspace(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.linspace(1, 200, 200, dtype=dtype).reshape(1, 200)
        b = xp.linspace(1, 200, 200, dtype=dtype).reshape(200, 1)
        return scp.special.betaln(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_betaln_scalar(self, xp, scp, dtype):
        import scipy.special  # NOQA

        return scp.special.betaln(dtype(2.5), dtype(3.5))

    def test_betaln_specific_vals(self):
        # specific values borrowed from SciPy test suite
        special = cupyx.scipy.special
        assert special.beta(1, 1) == 0.0
        testing.assert_allclose(special.beta(-100.3, 1e-200),
                                special.gammaln(1e-200))
        testing.assert_allclose(special.beta(0.0342, 171),
                                3.1811881124242447, rtol=1e-14, atol=0)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-2, rtol=1e-3, scipy_name='scp')
    def test_betaln_inf_and_nan(self, xp, scp, dtype):
        import scipy.special  # NOQA

        a = xp.array([-numpy.inf, numpy.nan, numpy.inf, 0], dtype=dtype)
        return scp.special.betaln(a.reshape((1, 4)), a.reshape((4, 1)))
