import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.special


class _TestBase:

    def test_ndtr(self):
        self.check_unary_linspace0_1('ndtr')

    def test_log_ndtr(self):
        self.check_unary_linspace0_1('log_ndtr')

    def test_ndtri(self):
        self.check_unary_linspace0_1('ndtri')

    def test_logit(self):
        self.check_unary_lower_precision('logit')

    def test_expit(self):
        self.check_unary_lower_precision('expit')

    @testing.with_requires('scipy>=1.8.0rc0')
    def test_log_expit(self):
        self.check_unary_lower_precision('log_expit')


atol = {'default': 1e-14, cupy.float64: 1e-14}
rtol = {'default': 1e-5, cupy.float64: 1e-14}

# not all functions pass at the stricter tolerances above
atol_low = {'default': 5e-4, cupy.float64: 1e-12}
rtol_low = {'default': 5e-4, cupy.float64: 1e-12}


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial(_TestBase):

    def _check_unary(self, a, name, scp):
        import scipy.special  # NOQA

        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def check_unary_lower_precision(self, name, xp, scp, dtype):
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary_linspace0_1(self, name, xp, scp, dtype):
        p = xp.linspace(0, 1, 1000, dtype=dtype)
        return self._check_unary(p, name, scp)

    def test_logit_nonfinite(self):
        logit = cupyx.scipy.special.logit
        assert float(logit(0)) == -numpy.inf
        assert float(logit(1)) == numpy.inf
        assert numpy.isnan(float(logit(1.1)))
        assert numpy.isnan(float(logit(-0.1)))

    @pytest.mark.parametrize('inverse', [False, True],
                             ids=['boxcox', 'inv_boxcox'])
    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def test_boxcox(self, xp, scp, dtype, inverse):
        import scipy.special  # NOQA

        # outputs are only finite over range (0, 1)
        x = xp.linspace(0.001, 1000, 1000, dtype=dtype).reshape((1, 1000))
        lmbda = xp.asarray([-5, 0, 5], dtype=dtype).reshape((3, 1))
        result = scp.special.boxcox(x, lmbda)
        if inverse:
            result = scp.special.inv_boxcox(result, lmbda)
        return result

    def test_boxcox_nonfinite(self):
        boxcox = cupyx.scipy.special.boxcox
        assert float(boxcox(0, -5)) == -numpy.inf
        assert numpy.isnan(float(boxcox(-0.1, 5)))

    @pytest.mark.parametrize('inverse', [False, True],
                             ids=['boxcox', 'inv_boxcox'])
    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def test_boxcox1p(self, xp, scp, dtype, inverse):
        import scipy.special  # NOQA

        x = xp.linspace(-0.99, 1000, 1000, dtype=dtype).reshape((1, 1000))
        lmbda = xp.asarray([-5, 0, 5], dtype=dtype).reshape((3, 1))
        result = scp.special.boxcox1p(x, lmbda)
        if inverse:
            result = scp.special.inv_boxcox1p(result, lmbda)
        return result

    def test_boxcox1p_nonfinite(self):
        boxcox1p = cupyx.scipy.special.boxcox1p
        assert float(boxcox1p(-1, -5)) == -numpy.inf
        assert numpy.isnan(float(boxcox1p(-1.1, 5)))


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(_TestBase):

    def _check_unary(self, a, name, scp):
        import scipy.special  # NOQA

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol_low, rtol=rtol_low,
                                 scipy_name='scp')
    def check_unary_lower_precision(self, name, xp, scp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return self._check_unary(a, name, scp)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=atol, rtol=rtol, scipy_name='scp')
    def check_unary_linspace0_1(self, name, xp, scp, dtype):
        a = xp.linspace(0, 1, 1000, dtype)
        return self._check_unary(a, name, scp)


class _TestDistributionsBase:

    def _test_scalar(self, function, args, expected, rtol=1e-12, atol=1e-12):
        special = cupyx.scipy.special
        function = getattr(special, function)
        testing.assert_allclose(function(*args), expected, rtol=rtol,
                                atol=atol)


@testing.gpu
@testing.with_requires('scipy')
class TestTwoArgumentDistribution(_TestDistributionsBase):

    @pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                        reason="avoid failures observed on HIP")
    @pytest.mark.parametrize('function', ['chdtr', 'chdtrc', 'chdtri',
                                          'pdtr', 'pdtrc', 'pdtri'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace_broadcast(self, xp, scp, dtype, function):
        import scipy.special  # NOQA

        func = getattr(scp.special, function)
        # chdtr* comparisons fail at < 1 degree of freedom
        minval = 1 if function.startswith('chdtr') else -1
        v = xp.arange(minval, 10, dtype=dtype)[:, xp.newaxis]
        if function in ['chdtri', 'pdtri']:
            # concentrate values around probability range ([0, 1])
            x = xp.linspace(-.1, 1.3, 20, dtype=dtype)
        else:
            # concentrate mostly on valid, positive values
            x = xp.linspace(-1, 10, 20, dtype=dtype)
        return func(v, x[xp.newaxis, :])

    @testing.for_float_dtypes()
    @testing.for_int_dtypes(name='int_dtype', no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace_pdtri(self, xp, scp, int_dtype, dtype):
        import scipy.special  # NOQA

        func = getattr(scp.special, 'pdtri')
        k = xp.arange(1, 10, dtype=int_dtype)[:, xp.newaxis]
        y = xp.linspace(0, 1, 20, dtype=dtype)[xp.newaxis, :]
        return func(k, y)

    @pytest.mark.parametrize(
        'function, args, expected',
        [('chdtr', (1, 0), 0.0),
         ('chdtr', (0.7, cupy.inf), 1.0),
         ('chdtr', (0.6, 3), 0.957890536704110),
         ('chdtrc', (1, 0), 1.0),
         ('chdtrc', (0.6, 3), 1 - 0.957890536704110),
         ('chdtri', (1, 1), 0.0),
         ('chdtri', (0.6, 1 - 0.957890536704110), 3),
         ]
    )
    def test_scalar(self, function, args, expected):
        self._test_scalar(function, args, expected)


@testing.gpu
@testing.with_requires('scipy')
class TestThreeArgumentDistributions(_TestDistributionsBase):

    @pytest.mark.parametrize('function', ['btdtr', 'btdtri', 'fdtr', 'fdtrc',
                                          'fdtri', 'gdtr', 'gdtrc'])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_linspace_broadcasted(self, xp, scp, dtype, function):
        """Linspace with three arguments.

        This method uses first two arguments with mostly non-negative values.
        In some cases, the last argument is constrained to range [0, 1]
        """

        import scipy.special  # NOQA

        func = getattr(scp.special, function)
        # a and b should be positive
        a = xp.linspace(-1, 21, 30, dtype=dtype)[:, xp.newaxis, xp.newaxis]
        b = xp.linspace(-1, 21, 30, dtype=dtype)[xp.newaxis, :, xp.newaxis]
        if function in ['fdtri', 'btdtr', 'btdtri']:
            # x should be in [0, 1] so concentrate values around that
            x = xp.linspace(-0.1, 1.3, 20, dtype=dtype)
        else:
            # x should be non-negative, but test with at least 1 negative value
            x = xp.linspace(-1, 10, 20, dtype=dtype)
        x = x[xp.newaxis, xp.newaxis, :]
        return func(a, b, x)

    # omit test with scipy < 1.5 due to change in ufunc type signatures
    @pytest.mark.parametrize('function', ['bdtr', 'bdtrc', 'bdtri'])
    @testing.for_float_dtypes()
    @testing.for_signed_dtypes(name='int_dtype')
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    @testing.with_requires('scipy>=1.5.0')
    @pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                        reason="avoid failures observed on HIP")
    def test_binomdist_linspace(self, xp, scp, function, dtype, int_dtype):
        import scipy.special  # NOQA

        # Skip cases deprecated in SciPy 1.5+ via this Cython code:
        # https://github.com/scipy/scipy/blob/cdb9b034d46c7ba0cacf65a9b2848c5d49c286c4/scipy/special/_legacy.pxd#L39-L43  # NOQA
        # All type casts except `dld->d` should raise a DeprecationWawrning
        # However on the SciPy side, this shows up as a SystemError
        #    SystemError: <class 'DeprecationWarning'> returned a result with an exception set  # NOQA
        safe_cast = xp.result_type(int_dtype, 'l') == xp.dtype('l')
        safe_cast &= xp.result_type(int_dtype, dtype) == xp.float64
        if not safe_cast:
            return xp.zeros((1,))

        func = getattr(scp.special, function)
        n = xp.linspace(0, 80, 80, dtype=int_dtype)[xp.newaxis, :, xp.newaxis]

        # broadcast to create k <= n
        k = xp.linspace(0, 1, 10, dtype=dtype)
        k = k[:, xp.newaxis, xp.newaxis] * n
        p = xp.linspace(0, 1, 5, dtype=dtype)[xp.newaxis, xp.newaxis, :]
        return func(k, n, p)

    @pytest.mark.parametrize('function', ['nbdtr', 'nbdtrc', 'nbdtri'])
    @testing.for_float_dtypes()
    @testing.for_signed_dtypes(name='int_dtype')
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_negbinomdist_linspace(self, xp, scp, function, dtype, int_dtype):
        import scipy.special  # NOQA

        func = getattr(scp.special, function)
        n = xp.linspace(0, 20, 20, dtype=int_dtype)[xp.newaxis, :, xp.newaxis]
        k = xp.linspace(0, 20, 20, dtype=int_dtype)[:, xp.newaxis, xp.newaxis]
        p = xp.linspace(0, 1, 5, dtype=dtype)[xp.newaxis, xp.newaxis, :]
        return func(k, n, p)

    @pytest.mark.parametrize(
        'function, args, expected',
        [('btdtr', (1, 1, 1), 1.0),
         ('btdtri', (1, 1, 1), 1.0),
         ('betainc', (1, 1, 0), 0.0),
         # Computed using Wolfram Alpha: CDF[FRatioDistribution[1e-6, 5], 10]
         ('fdtr', (1e-6, 5, 10), 0.9999940790193488),
         ('fdtrc', (1, 1, 0), 1.0),
         # Computed using Wolfram Alpha:
         #   1 - CDF[FRatioDistribution[2, 1/10], 1e10]
         ('fdtrc', (2, 0.1, 1e10), 0.2722378462129351),
         # From Wolfram Alpha:
         #   CDF[FRatioDistribution[1/10, 1], 3] = 0.8756751669632106...
         ('fdtri', (0.1, 1, 0.8756751669632106), 3.0),
         ('gdtr', (1, 1, 0), 0.0),
         ('gdtr', (1, 1, cupy.inf), 1.0),
         ('gdtrc', (1, 1, 0), 1.0),
         ('bdtr', (1, 1, 0.5), 1.0),
         ('bdtrc', (1, 3, 0.5), 0.5),
         ('bdtri', (1, 3, 0.5), 0.5),
         ('nbdtr', (1, 1, 1), 1.0),
         ('nbdtrc', (1, 1, 1), 0.0),
         ('nbdtri', (1, 1, 1), 1.0),
         ]
    )
    def test_scalar(self, function, args, expected):
        self._test_scalar(function, args, expected)
