import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.special


class _TestBase:

    def test_ndtr(self):
        self.check_unary_linspace0_1('ndtr')

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

    def _test_scalar(self, function, args, expected, rtol=None):
        special = cupyx.scipy.special
        function = getattr(special, function)
        if rtol is None:
            testing.assert_array_equal(function(*args), expected)
        else:
            testing.assert_allclose(function(*args), expected, rtol=rtol)


@testing.gpu
@testing.with_requires('scipy')
class TestTwoArgumentDistribution(_TestDistributionsBase):

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
        'function, args, expected, rtol',
        [('chdtr', (1, 0), 0.0, None),
         ('chdtr', (0.7, cupy.inf), 1.0, None),
         ('chdtr', (0.6, 3), 0.957890536704110, 1e-12),
         ('chdtrc', (1, 0), 1.0, None),
         ('chdtrc', (0.6, 3), 1 - 0.957890536704110, 1e-12),
         ('chdtri', (1, 1), 0.0, None),
         ('chdtri', (0.6, 1 - 0.957890536704110), 3, 1e-12),
         ]
    )
    def test_scalar(self, function, args, expected, rtol):
        self._test_scalar(function, args, expected, rtol)


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

    @pytest.mark.parametrize('function', ['bdtr', 'bdtrc', 'bdtri'])
    @testing.for_float_dtypes()
    @testing.for_signed_dtypes(name='int_dtype')
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_binomdist_linspace(self, xp, scp, function, dtype, int_dtype):
        import scipy.special  # NOQA

        if xp.dtype(int_dtype) not in [xp.int32, xp.int64]:
            if xp.dtype(dtype) != xp.float64:
                # Skip cases deprecated in SciPy 1.7+ via this Cython code:
                # https://github.com/scipy/scipy/blob/cdb9b034d46c7ba0cacf65a9b2848c5d49c286c4/scipy/special/_legacy.pxd#L39-L43  # NOQA
                # It causes infinite recursion in numpy_cupy_allclose
                return xp.zeros((1,))

        func = getattr(scp.special, function)
        n = xp.linspace(0, 80, 80, dtype=int_dtype)[xp.newaxis, :, xp.newaxis]
        # broadcast to create k <= n
        k = xp.linspace(0, 1, 10, dtype=dtype)[:, xp.newaxis, xp.newaxis] * n
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
        'function, args, expected, rtol',
        [('btdtr', (1, 1, 1), 1.0, None),
         ('btdtri', (1, 1, 1), 1.0, None),
         ('betainc', (1, 1, 0), 0.0, None),
         # Computed using Wolfram Alpha: CDF[FRatioDistribution[1e-6, 5], 10]
         ('fdtr', (1e-6, 5, 10), 0.9999940790193488, 1e-12),
         ('fdtrc', (1, 1, 0), 1.0, None),
         # Computed using Wolfram Alpha:
         #   1 - CDF[FRatioDistribution[2, 1/10], 1e10]
         ('fdtrc', (2, 0.1, 1e10), 0.2722378462129351, 1e-12),
         # From Wolfram Alpha:
         #   CDF[FRatioDistribution[1/10, 1], 3] = 0.8756751669632106...
         ('fdtri', (0.1, 1, 0.8756751669632106), 3.0, 1e-12),
         ('gdtr', (1, 1, 0), 0.0, None),
         ('gdtr', (1, 1, cupy.inf), 1.0, None),
         ('gdtrc', (1, 1, 0), 1.0, None),
         ('bdtr', (1, 1, 0.5), 1.0, None),
         ('bdtrc', (1, 3, 0.5), 0.5, None),
         ('bdtri', (1, 3, 0.5), 0.5, None),
         ('nbdtr', (1, 1, 1), 1.0, None),
         ('nbdtrc', (1, 1, 1), 0.0, None),
         ('nbdtri', (1, 1, 1), 1.0, None),
         ]
    )
    def test_scalar(self, function, args, expected, rtol):
        self._test_scalar(function, args, expected, rtol)
