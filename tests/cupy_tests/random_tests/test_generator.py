import functools
import os
import threading
import unittest

import numpy
import six

import cupy
from cupy import core
from cupy import cuda
from cupy.random import generator
from cupy import testing
from cupy.testing import attr
from cupy.testing import condition
from cupy.testing import hypothesis


def numpy_cupy_equal_continuous_distribution(significance_level, name='xp'):
    """Decorator that tests the distributions of NumPy samples and CuPy ones.

    Args:
        significance_level (float): The test fails if p-value is lower than
            this argument.
        name(str): Argument name whose value is either
            ``numpy`` or ``cupy`` module.

    Decorated test fixture is required to return samples from the same
    distribution even if ``xp`` is ``numpy`` or ``cupy``.

    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            cupy_result = impl(self, *args, **kw)

            kw[name] = numpy
            numpy_result = impl(self, *args, **kw)

            self.assertIsNotNone(cupy_result)
            self.assertIsNotNone(numpy_result)
            d_plus, d_minus, p_value = \
                two_sample_Kolmogorov_Smirnov_test(
                    cupy.asnumpy(cupy_result), numpy_result)
            if p_value < significance_level:
                message = '''Rejected null hypothesis:
p: %f
D+ (cupy < numpy): %f
D- (cupy > numpy): %f''' % (p_value, d_plus, d_minus)
                raise AssertionError(message)
        return test_func
    return decorator


def two_sample_Kolmogorov_Smirnov_test(observed1, observed2):
    """Computes the Kolmogorov-Smirnov statistic on 2 samples

    Unlike `scipy.stats.ks_2samp`, the returned p-value is not accurate
    for large p.
    """
    assert observed1.dtype == observed2.dtype
    n1, = observed1.shape
    n2, = observed2.shape
    assert n1 >= 100 and n2 >= 100
    observed = numpy.concatenate([observed1, observed2])
    indices = numpy.argsort(observed)
    observed = observed[indices]  # sort
    ds = numpy.cumsum(numpy.where(indices < n1, -n2, n1).astype(numpy.int64))
    assert ds[-1] == 0
    check = numpy.concatenate([observed[:-1] < observed[1:], [True]])
    ds = ds[check]
    d_plus = float(ds.max()) / (n1 * n2)
    d_minus = -float(ds.min()) / (n1 * n2)
    d = max(d_plus, d_minus)
    # Approximate p = special.kolmogorov(d * numpy.sqrt(n1 * n2 / (n1 + n2)))
    p = min(1.0, 2.0 * numpy.exp(-2.0 * d**2 * n1 * n2 / (n1 + n2)))
    return d_plus, d_minus, p


class RandomGeneratorTestCase(unittest.TestCase):

    target_method = None

    def setUp(self):
        self.__seed = testing.generate_seed()
        self.rs = generator.RandomState(seed=self.__seed)

    def _get_generator_func(self, *args, **kwargs):
        assert isinstance(self.target_method, str), (
            'generate_method must be overridden')
        f = getattr(self.rs, self.target_method)
        return lambda: f(*args, **kwargs)

    def _generate_check_repro(self, func, seed):
        # Sample a random array while checking reproducibility
        self.rs.seed(seed)
        x = func()
        self.rs.seed(seed)
        y = func()
        testing.assert_array_equal(
            x, y,
            'Randomly generated arrays with the same seed did not match')
        return x

    def generate(self, *args, **kwargs):
        # Pick one sample from generator.
        # Reproducibility is checked by repeating seed-and-sample cycle twice.
        func = self._get_generator_func(*args, **kwargs)
        return self._generate_check_repro(func, self.__seed)

    def generate_many(self, *args, **kwargs):
        # Pick many samples from generator.
        # Reproducibility is checked only for the first sample,
        # because it's very slow to set seed every time.
        _count = kwargs.pop('_count', None)
        assert _count is not None, '_count is required'
        func = self._get_generator_func(*args, **kwargs)

        if _count == 0:
            return []

        vals = [self._generate_check_repro(func, self.__seed)]
        for _ in range(1, _count):
            vals.append(func())
        return vals

    def check_ks(self, significance_level, cupy_len=100, numpy_len=1000):
        return functools.partial(
            self._check_ks, significance_level, cupy_len, numpy_len)

    def _check_ks(
            self, significance_level, cupy_len, numpy_len,
            *args, **kwargs):
        assert 'size' in kwargs

        # cupy
        func = self._get_generator_func(*args, **kwargs)
        vals_cupy = func()
        assert vals_cupy.size > 0
        count = 1 + (cupy_len - 1) // vals_cupy.size
        vals_cupy = [vals_cupy]
        for _ in range(1, count):
            vals_cupy.append(func())
        vals_cupy = cupy.stack(vals_cupy).ravel()

        # numpy
        kwargs['size'] = numpy_len
        dtype = kwargs.pop('dtype', None)
        numpy_rs = numpy.random.RandomState(self.__seed)
        vals_numpy = getattr(numpy_rs, self.target_method)(*args, **kwargs)
        if dtype is not None:
            vals_numpy = vals_numpy.astype(dtype, copy=False)

        # test
        d_plus, d_minus, p_value = \
            two_sample_Kolmogorov_Smirnov_test(
                cupy.asnumpy(vals_cupy), vals_numpy)
        if p_value < significance_level:
            message = '''Rejected null hypothesis:
p: %f
D+ (cupy < numpy): %f
D- (cupy > numpy): %f''' % (p_value, d_plus, d_minus)
            raise AssertionError(message)


def _xp_random(xp, method_name):
    method = getattr(xp.random.RandomState(), method_name)
    if xp == cupy:
        return method

    def f(*args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        ret = method(*args, **kwargs)
        if dtype is not None:
            ret = ret.astype(dtype, copy=False)
        return ret

    return f


@testing.fix_random()
@testing.gpu
class TestRandomState(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def check_seed(self, seed):
        rs = self.rs

        rs.seed(seed)
        xs1 = [rs.uniform() for _ in range(100)]

        rs.seed(seed)
        xs2 = [rs.uniform() for _ in range(100)]

        rs.seed(seed)
        rs.seed(None)
        xs3 = [rs.uniform() for _ in range(100)]

        # Random state must be reproducible
        assert xs1 == xs2
        # Random state must be initialized randomly with seed=None
        assert xs1 != xs3

    @testing.for_int_dtypes()
    def test_seed_not_none(self, dtype):
        self.check_seed(dtype(0))

    @testing.for_dtypes([numpy.complex_])
    def test_seed_invalid_type_complex(self, dtype):
        with self.assertRaises(TypeError):
            self.rs.seed(dtype(0))

    @testing.for_float_dtypes()
    def test_seed_invalid_type_float(self, dtype):
        with self.assertRaises(TypeError):
            self.rs.seed(dtype(0))

    def test_array_seed(self):
        self.check_seed(numpy.random.randint(0, 2**31, size=40))


@testing.parameterize(
    {'a': 1.0, 'b': 3.0},
    {'a': 3.0, 'b': 3.0},
    {'a': 3.0, 'b': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestBeta(RandomGeneratorTestCase):

    target_method = 'beta'

    def test_beta(self):
        self.generate(a=self.a, b=self.b, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_beta_ks(self, dtype):
        self.check_ks(0.05)(
            a=self.a, b=self.b, size=2000, dtype=dtype)


@testing.parameterize(
    {'n': 5, 'p': 0.5},
    {'n': 5, 'p': 0.0},
    {'n': 5, 'p': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestBinomial(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'binomial'

    def test_binomial(self):
        self.generate(n=self.n, p=self.p, size=(3, 2))


@testing.parameterize(
    {'df': 1.0},
    {'df': 3.0},
    {'df': 10.0},
)
@testing.gpu
@testing.fix_random()
class TestChisquare(RandomGeneratorTestCase):

    target_method = 'chisquare'

    def test_chisquare(self):
        self.generate(df=self.df, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_chisquare_ks(self, dtype):
        self.check_ks(0.05)(
            df=self.df, size=2000, dtype=dtype)


@testing.gpu
@testing.parameterize(
    {'alpha': cupy.array([1.0, 1.0, 1.0])},
    {'alpha': cupy.array([1.0, 3.0, 5.0])},
)
@testing.fix_random()
class TestDirichlet(RandomGeneratorTestCase):

    target_method = 'dirichlet'

    def test_dirichlet(self):
        self.generate(alpha=self.alpha, size=(3, 2, 3))

    # TODO(kataoka): add distribution test


@testing.parameterize(
    {'scale': 1.0},
    {'scale': 3.0},
    {'scale': 10.0},
)
@testing.gpu
@testing.fix_random()
class TestExponential(RandomGeneratorTestCase):

    target_method = 'exponential'

    def test_exponential(self):
        self.generate(scale=self.scale, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_exponential_ks(self, dtype):
        self.check_ks(0.05)(
            self.scale, size=2000, dtype=dtype)


@testing.parameterize(
    {'dfnum': 1.0, 'dfden': 3.0},
    {'dfnum': 3.0, 'dfden': 3.0},
    {'dfnum': 3.0, 'dfden': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestF(RandomGeneratorTestCase):

    target_method = 'f'

    def test_f(self):
        self.generate(dfnum=self.dfnum, dfden=self.dfden, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_f_ks(self, dtype):
        self.check_ks(0.05)(
            self.dfnum, self.dfden, size=2000, dtype=dtype)


@testing.parameterize(
    {'shape': 0.5, 'scale': 0.5},
    {'shape': 1.0, 'scale': 0.5},
    {'shape': 3.0, 'scale': 0.5},
    {'shape': 0.5, 'scale': 1.0},
    {'shape': 1.0, 'scale': 1.0},
    {'shape': 3.0, 'scale': 1.0},
    {'shape': 0.5, 'scale': 3.0},
    {'shape': 1.0, 'scale': 3.0},
    {'shape': 3.0, 'scale': 3.0},
)
@testing.gpu
@testing.fix_random()
class TestGamma(RandomGeneratorTestCase):

    target_method = 'gamma'

    def test_gamma_1(self):
        self.generate(shape=self.shape, scale=self.scale, size=(3, 2))

    def test_gamma_2(self):
        self.generate(shape=self.shape, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_gamma_ks(self, dtype):
        self.check_ks(0.05)(
            self.shape, self.scale, size=2000, dtype=dtype)


@testing.parameterize(
    {'p': 0.5},
    {'p': 0.1},
    {'p': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestGeometric(RandomGeneratorTestCase):

    target_method = 'geometric'

    def test_geometric(self):
        self.generate(p=self.p, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_geometric_ks(self, dtype):
        self.check_ks(0.05)(
            p=self.p, size=2000, dtype=dtype)


@testing.parameterize(
    {'ngood': 1, 'nbad': 1, 'nsample': 1},
    {'ngood': 1, 'nbad': 1, 'nsample': 2},
)
@testing.gpu
@testing.fix_random()
class TestHypergeometric(RandomGeneratorTestCase):

    target_method = 'hypergeometric'

    def test_hypergeometric(self):
        self.generate(ngood=self.ngood, nbad=self.nbad, nsample=self.nsample,
                      size=(3, 2))

    # TODO(kataoka): add distribution test


@testing.gpu
@testing.fix_random()
class TestLaplace(RandomGeneratorTestCase):

    target_method = 'laplace'

    def test_laplace_1(self):
        self.generate()

    def test_laplace_2(self):
        self.generate(0.0, 1.0, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_laplace_ks_1(self, dtype):
        self.check_ks(0.05)(
            size=2000, dtype=dtype)

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_laplace_ks_2(self, dtype):
        self.check_ks(0.05)(
            2.3, 4.5, size=2000, dtype=dtype)


@testing.gpu
@testing.fix_random()
class TestLogistic(RandomGeneratorTestCase):

    target_method = 'logistic'

    def test_logistic_1(self):
        self.generate()

    def test_logistic_2(self):
        self.generate(0.0, 1.0, size=(3, 2))

    @attr.slow
    @condition.repeat(10)
    def test_standard_logistic_isfinite(self):
        x = self.generate(size=10**7)
        self.assertTrue(cupy.isfinite(x).all())

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_logistic_ks_1(self, dtype):
        self.check_ks(0.05)(
            size=2000, dtype=dtype)

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_logistic_ks_2(self, dtype):
        self.check_ks(0.05)(
            2.3, 4.5, size=2000, dtype=dtype)


@testing.gpu
@testing.parameterize(*[
    {'args': (0.0, 1.0), 'size': None},
    {'args': (10.0, 20.0), 'size': None},
    {'args': (0.0, 1.0), 'size': 10},
    {'args': (0.0, 1.0), 'size': (1, 2, 3)},
    {'args': (0.0, 1.0), 'size': 3},
    {'args': (0.0, 1.0), 'size': (3, 3)},
    {'args': (0.0, 1.0), 'size': ()},
])
@testing.fix_random()
class TestLogNormal(RandomGeneratorTestCase):

    target_method = 'lognormal'

    def check_lognormal(self, dtype):
        vals = self.generate_many(
            self.args[0], self.args[1], self.size, dtype, _count=10)

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape
            assert (0 <= val).all()

    def test_lognormal_float(self):
        self.check_lognormal(float)

    def test_lognormal_float32(self):
        self.check_lognormal(numpy.float32)

    def test_lognormal_float64(self):
        self.check_lognormal(numpy.float64)

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_lognormal_ks(self, dtype):
        self.check_ks(0.05)(
            *self.args, size=self.size, dtype=dtype)


@testing.parameterize(
    {'p': 0.5},
    {'p': 0.1},
    {'p': 0.9},
)
@testing.gpu
@testing.fix_random()
class TestLogseries(RandomGeneratorTestCase):

    target_method = 'logseries'

    def test_logseries(self):
        self.generate(p=self.p, size=(3, 2))

    # TODO(kataoka): add distribution test


@testing.gpu
@testing.parameterize(*[
    {'args': ([0., 0.], [[1., 0.], [0., 1.]]), 'size': None, 'tol': 1e-6},
    {'args': ([10., 10.], [[20., 10.], [10., 20.]]),
     'size': None, 'tol': 1e-6},
    {'args': ([0., 0.], [[1., 0.], [0., 1.]]), 'size': 10, 'tol': 1e-6},
    {'args': ([0., 0.], [[1., 0.], [0., 1.]]), 'size': (1, 2, 3), 'tol': 1e-6},
    {'args': ([0., 0.], [[1., 0.], [0., 1.]]), 'size': 3, 'tol': 1e-6},
    {'args': ([0., 0.], [[1., 0.], [0., 1.]]), 'size': (3, 3), 'tol': 1e-6},
    {'args': ([0., 0.], [[1., 0.], [0., 1.]]), 'size': (), 'tol': 1e-6},
])
@testing.fix_random()
class TestMultivariateNormal(RandomGeneratorTestCase):

    target_method = 'multivariate_normal'

    def check_multivariate_normal(self, dtype):
        vals = self.generate_many(
            mean=self.args[0], cov=self.args[1], size=self.size, tol=self.tol,
            dtype=dtype, _count=10)

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape + (2,)

    def test_multivariate_normal_float32(self):
        self.check_multivariate_normal(numpy.float32)

    def test_multivariate_normal_float64(self):
        self.check_multivariate_normal(numpy.float64)

    # TODO(kataoka): add distribution test


@testing.parameterize(
    {'n': 5, 'p': 0.5},
)
@testing.gpu
@testing.fix_random()
class TestNegativeBinomial(RandomGeneratorTestCase):
    target_method = 'negative_binomial'

    def test_negative_binomial(self):
        self.generate(n=self.n, p=self.p, size=(3, 2))

    # TODO(kataoka): add distribution test


@testing.parameterize(
    {'df': 1.5, 'nonc': 2.0},
    {'df': 2.0, 'nonc': 0.0},
)
@testing.gpu
@testing.with_requires('numpy>=1.11')
@testing.fix_random()
class TestNoncentralChisquare(RandomGeneratorTestCase):

    target_method = 'noncentral_chisquare'

    def test_noncentral_chisquare(self):
        self.generate(df=self.df, nonc=self.nonc, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_noncentral_chisquare_ks(self, dtype):
        self.check_ks(0.05)(
            self.df, self.nonc, size=2000, dtype=dtype)


@testing.parameterize(
    {'dfnum': 2.0, 'dfden': 3.0, 'nonc': 4.0},
    {'dfnum': 2.5, 'dfden': 1.5, 'nonc': 0.0},
)
@testing.gpu
@testing.fix_random()
class TestNoncentralF(RandomGeneratorTestCase):

    target_method = 'noncentral_f'

    def test_noncentral_f(self):
        self.generate(
            dfnum=self.dfnum, dfden=self.dfden, nonc=self.nonc, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_noncentral_f_ks(self, dtype):
        self.check_ks(0.05)(
            self.dfnum, self.dfden, self.nonc, size=2000, dtype=dtype)


@testing.gpu
@testing.parameterize(*[
    {'args': (0.0, 1.0), 'size': None},
    {'args': (10.0, 20.0), 'size': None},
    {'args': (0.0, 1.0), 'size': 10},
    {'args': (0.0, 1.0), 'size': (1, 2, 3)},
    {'args': (0.0, 1.0), 'size': 3},
    {'args': (0.0, 1.0), 'size': (3, 3)},
    {'args': (0.0, 1.0), 'size': ()},
])
@testing.fix_random()
class TestNormal(RandomGeneratorTestCase):

    target_method = 'normal'

    def check_normal(self, dtype):
        vals = self.generate_many(
            self.args[0], self.args[1], self.size, dtype, _count=10)

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape

    def test_normal_float32(self):
        self.check_normal(numpy.float32)

    def test_normal_float64(self):
        self.check_normal(numpy.float64)

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_normal_ks(self, dtype):
        self.check_ks(0.05)(
            *self.args, size=self.size, dtype=dtype)


@testing.parameterize(
    {'a': 1.0},
    {'a': 3.0},
    {'a': 10.0},
)
@testing.gpu
@testing.fix_random()
class TestPareto(RandomGeneratorTestCase):

    target_method = 'pareto'

    def test_pareto(self):
        self.generate(a=self.a, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_pareto_ks(self, dtype):
        self.check_ks(0.05)(
            a=self.a, size=2000, dtype=dtype)


@testing.parameterize(
    {'lam': 1.0},
    {'lam': 3.0},
    {'lam': 10.0},
)
@testing.gpu
@testing.fix_random()
class TestPoisson(RandomGeneratorTestCase):

    target_method = 'poisson'

    def test_poisson(self):
        self.generate(lam=self.lam, size=(3, 2))

    # TODO(kataoka): add distribution test


@testing.parameterize(
    {'df': 1.0},
    {'df': 3.0},
    {'df': 10.0},
)
@testing.gpu
@testing.fix_random()
class TestStandardT(RandomGeneratorTestCase):

    target_method = 'standard_t'

    def test_standard_t(self):
        self.generate(df=self.df, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_standard_t_ks(self, dtype):
        self.check_ks(0.05)(
            df=self.df, size=2000, dtype=dtype)


@testing.gpu
@testing.parameterize(*[
    {'size': None},
    {'size': 10},
    {'size': (1, 2, 3)},
    {'size': 3},
    {'size': ()},
])
@testing.fix_random()
class TestRandomSample(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def check_random_sample(self, dtype):
        vals = [self.rs.random_sample(self.size, dtype) for _ in range(10)]

        shape = core.get_size(self.size)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype == dtype
            assert val.shape == shape
            assert (0 <= val).all()
            assert (val < 1).all()

    def test_random_sample_float32(self):
        self.check_random_sample(numpy.float32)

    def test_random_sample_float64(self):
        self.check_random_sample(numpy.float64)


@testing.fix_random()
class TestRandomSampleDistrib(unittest.TestCase):

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    @numpy_cupy_equal_continuous_distribution(0.05)
    def test_random_sample_ks(self, xp, dtype):
        return _xp_random(xp, 'random_sample')(size=2000, dtype=dtype)


@testing.fix_random()
@testing.gpu
class TestRandAndRandN(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def test_rand_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.rand(1, 2, 3, unnecessary='unnecessary_argument')

    def test_randn_invalid_argument(self):
        with self.assertRaises(TypeError):
            self.rs.randn(1, 2, 3, unnecessary='unnecessary_argument')


@testing.parameterize(
    {'a': 0.5},
)
@testing.gpu
@testing.fix_random()
class TestPower(RandomGeneratorTestCase):

    target_method = 'power'

    def test_power(self):
        self.generate(a=self.a, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_power_ks(self, dtype):
        self.check_ks(0.05)(
            a=self.a, size=2000, dtype=dtype)


@testing.parameterize(
    {'scale': 1.0},
    {'scale': 3.0},
)
@testing.gpu
@testing.fix_random()
class TestRayleigh(RandomGeneratorTestCase):

    target_method = 'rayleigh'

    def test_rayleigh(self):
        self.generate(scale=self.scale, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_rayleigh_ks(self, dtype):
        self.check_ks(0.05)(
            scale=self.scale, size=2000, dtype=dtype)


@testing.gpu
@testing.fix_random()
class TestStandardCauchy(RandomGeneratorTestCase):

    target_method = 'standard_cauchy'

    def test_standard_cauchy(self):
        self.generate(size=(3, 2))

    @attr.slow
    @condition.repeat(10)
    def test_standard_cauchy_isfinite(self):
        x = self.generate(size=10**7)
        self.assertTrue(cupy.isfinite(x).all())

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_standard_cauchy_ks(self, dtype):
        self.check_ks(0.05)(
            size=2000, dtype=dtype)


@testing.parameterize(
    {'shape': 0.5},
    {'shape': 1.0},
    {'shape': 3.0},
)
@testing.gpu
@testing.fix_random()
class TestStandardGamma(RandomGeneratorTestCase):

    target_method = 'standard_gamma'

    def test_standard_gamma(self):
        self.generate(shape=self.shape, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_standard_gamma_ks(self, dtype):
        self.check_ks(0.05)(
            shape=self.shape, size=2000, dtype=dtype)


@testing.fix_random()
@testing.gpu
class TestInterval(RandomGeneratorTestCase):

    target_method = '_interval'

    def test_zero(self):
        shape = (2, 3)
        vals = self.generate_many(0, shape, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype.kind in 'iu'
            assert val.shape == shape
            assert (val == 0).all()

    def test_shape_zero(self):
        mx = 10
        vals = self.generate_many(mx, None, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype.kind in 'iu'
            assert val.shape == ()
            assert (0 <= val).all()
            assert (val <= mx).all()
        # TODO(niboshi): Distribution test

    def test_shape_one_dim(self):
        mx = 10
        size = 20
        vals = self.generate_many(mx, size, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype.kind in 'iu'
            assert val.shape == (size,)
            assert (0 <= val).all()
            assert (val <= mx).all()
        # TODO(niboshi): Distribution test

    def test_shape_multi_dim(self):
        mx = 10
        shape = (1, 2)
        vals = self.generate_many(mx, shape, _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype.kind in 'iu'
            assert val.shape == shape
            assert (0 <= val).all()
            assert (val <= mx).all()
        # TODO(niboshi): Distribution test

    def test_bound_1(self):
        vals = self.generate_many(10, (2, 3), _count=10)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype.kind in 'iu'
            assert val.shape == (2, 3)
            assert (0 <= val).all()
            assert (val <= 10).all()

    def test_bound_2(self):
        vals = self.generate_many(2, None, _count=20)
        for val in vals:
            assert isinstance(val, cupy.ndarray)
            assert val.dtype.kind in 'iu'
            assert val.shape == ()
            assert (0 <= val).all()
            assert (val <= 2).all()

    @condition.repeat(3, 10)
    def test_goodness_of_fit(self):
        mx = 5
        trial = 100
        vals = self.generate_many(mx, None, _count=trial)
        vals = [val.get() for val in vals]
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 2))[0]
        expected = numpy.array([float(trial) / (mx + 1)] * (mx + 1))
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.repeat(3)
    def test_goodness_of_fit_2(self):
        mx = 5
        vals = self.generate(mx, (5, 5)).get()
        counts = numpy.histogram(vals, bins=numpy.arange(mx + 2))[0]
        expected = numpy.array([float(vals.size) / (mx + 1)] * (mx + 1))
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


@testing.fix_random()
@testing.gpu
class TestTomaxint(RandomGeneratorTestCase):

    target_method = 'tomaxint'

    def test_tomaxint_none(self):
        x = self.generate()
        self.assertEqual(x.shape, ())
        self.assertTrue((0 <= x).all())
        self.assertTrue((x <= cupy.iinfo(cupy.int_).max).all())

    def test_tomaxint_int(self):
        x = self.generate(3)
        self.assertEqual(x.shape, (3,))
        self.assertTrue((0 <= x).all())
        self.assertTrue((x <= cupy.iinfo(cupy.int_).max).all())

    def test_tomaxint_tuple(self):
        x = self.generate((2, 3))
        self.assertEqual(x.shape, (2, 3))
        self.assertTrue((0 <= x).all())
        self.assertTrue((x <= cupy.iinfo(cupy.int_).max).all())


@testing.parameterize(
    {'a': 3, 'size': 2, 'p': None},
    {'a': 3, 'size': 2, 'p': [0.3, 0.3, 0.4]},
    {'a': 3, 'size': (5, 5), 'p': [0.3, 0.3, 0.4]},
    {'a': 3, 'size': (5, 5), 'p': numpy.array([0.3, 0.3, 0.4])},
    {'a': 3, 'size': (), 'p': None},
    {'a': numpy.array([0.0, 1.0, 2.0]), 'size': 2, 'p': [0.3, 0.3, 0.4]},
)
@testing.fix_random()
@testing.gpu
class TestChoice1(RandomGeneratorTestCase):

    target_method = 'choice'

    def test_dtype_shape(self):
        v = self.generate(a=self.a, size=self.size, p=self.p)
        if isinstance(self.size, six.integer_types):
            expected_shape = (self.size,)
        else:
            expected_shape = self.size
        if isinstance(self.a, numpy.ndarray):
            expected_dtype = 'float'
        else:
            expected_dtype = 'int64'
        self.assertEqual(v.dtype, expected_dtype)
        self.assertEqual(v.shape, expected_shape)

    @condition.repeat(3, 10)
    def test_bound(self):
        vals = self.generate_many(
            a=self.a, size=self.size, p=self.p, _count=20)
        vals = [val.get() for val in vals]
        size_ = self.size if isinstance(self.size, tuple) else (self.size,)
        for val in vals:
            self.assertEqual(val.shape, size_)
        self.assertEqual(min(val.min() for val in vals), 0)
        self.assertEqual(max(val.max() for val in vals), 2)


@testing.parameterize(
    {'a': [0, 1, 2], 'size': 2, 'p': [0.3, 0.3, 0.4]},
)
@testing.fix_random()
@testing.gpu
class TestChoice2(RandomGeneratorTestCase):

    target_method = 'choice'

    def test_dtype_shape(self):
        v = self.generate(a=self.a, size=self.size, p=self.p)
        if isinstance(self.size, six.integer_types):
            expected_shape = (self.size,)
        else:
            expected_shape = self.size
        if isinstance(self.a, numpy.ndarray):
            expected_dtype = 'float'
        else:
            expected_dtype = 'int'
        self.assertEqual(v.dtype, expected_dtype)
        self.assertEqual(v.shape, expected_shape)

    @condition.repeat(3, 10)
    def test_bound(self):
        vals = self.generate_many(
            a=self.a, size=self.size, p=self.p, _count=20)
        vals = [val.get() for val in vals]
        size_ = self.size if isinstance(self.size, tuple) else (self.size,)
        for val in vals:
            self.assertEqual(val.shape, size_)
        self.assertEqual(min(val.min() for val in vals), 0)
        self.assertEqual(max(val.max() for val in vals), 2)


@testing.fix_random()
@testing.gpu
class TestChoiceChi(RandomGeneratorTestCase):

    target_method = 'choice'

    @condition.repeat(3, 10)
    def test_goodness_of_fit(self):
        trial = 100
        vals = self.generate_many(3, 1, True, [0.3, 0.3, 0.4], _count=trial)
        vals = [val.get() for val in vals]
        counts = numpy.histogram(vals, bins=numpy.arange(4))[0]
        expected = numpy.array([30, 30, 40])
        self.assertTrue(hypothesis.chi_square_test(counts, expected))

    @condition.repeat(3, 10)
    def test_goodness_of_fit_2(self):
        vals = self.generate(3, (5, 20), True, [0.3, 0.3, 0.4]).get()
        counts = numpy.histogram(vals, bins=numpy.arange(4))[0]
        expected = numpy.array([30, 30, 40])
        self.assertTrue(hypothesis.chi_square_test(counts, expected))


@testing.fix_random()
@testing.gpu
class TestChoiceMultinomial(unittest.TestCase):

    @condition.repeat(3, 10)
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(atol=0.02)
    def test_choice_multinomial(self, xp, dtype):
        p = xp.array([0.5, 0.25, 0.125, 0.125], dtype)
        trial = 10000
        x = xp.random.choice(len(p), trial, p=p)
        y = xp.bincount(x).astype('f') / trial
        return y


@testing.parameterize(
    {'a': 3.1, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': None, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': -3, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': [[0, 1], [2]], 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': [], 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': 3, 'size': 1, 'p': [[0.1, 0.1], [0.8]]},
    {'a': 2, 'size': 1, 'p': [0.1, 0.1, 0.8]},
    {'a': 3, 'size': 1, 'p': [-0.1, 0.3, 0.8]},
    {'a': 3, 'size': 1, 'p': [0.1, 0.1, 0.7]},
)
@testing.fix_random()
@testing.gpu
class TestChoiceFailure(unittest.TestCase):

    def setUp(self):
        self.rs = generator.RandomState(seed=testing.generate_seed())

    def test_choice_invalid_value(self):
        with self.assertRaises(ValueError):
            self.rs.choice(a=self.a, size=self.size, p=self.p)


@testing.parameterize(
    {'a': 5, 'size': 2},
    {'a': 5, 'size': (2, 2)},
    {'a': 5, 'size': ()},
    {'a': numpy.array([0.0, 2.0, 4.0]), 'size': 2},
)
@testing.fix_random()
@testing.gpu
class TestChoiceReplaceFalse(RandomGeneratorTestCase):

    target_method = 'choice'

    def test_dtype_shape(self):
        v = self.generate(a=self.a, size=self.size, replace=False)
        if isinstance(self.size, six.integer_types):
            expected_shape = (self.size,)
        else:
            expected_shape = self.size
        if isinstance(self.a, numpy.ndarray):
            expected_dtype = 'float'
        else:
            expected_dtype = 'int'
        self.assertEqual(v.dtype, expected_dtype)
        self.assertEqual(v.shape, expected_shape)

    @condition.repeat(3, 10)
    def test_bound(self):
        val = self.generate(a=self.a, size=self.size, replace=False).get()
        size = self.size if isinstance(self.size, tuple) else (self.size,)
        self.assertEqual(val.shape, size)
        self.assertTrue((0 <= val).all())
        self.assertTrue((val < 5).all())
        val = numpy.asarray(val)
        self.assertEqual(numpy.unique(val).size, val.size)


@testing.gpu
@testing.fix_random()
class TestGumbel(RandomGeneratorTestCase):

    target_method = 'gumbel'

    def test_gumbel_1(self):
        self.generate()

    def test_gumbel_2(self):
        self.generate(0.0, 1.0, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_gumbel_ks_1(self, dtype):
        self.check_ks(0.05)(
            size=2000, dtype=dtype)

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_gumbel_ks_2(self, dtype):
        self.check_ks(0.05)(
            2.3, 4.5, size=2000, dtype=dtype)


@testing.gpu
@testing.fix_random()
class TestRandint(RandomGeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'randint'

    def test_randint_1(self):
        self.generate(3)

    def test_randint_2(self):
        self.generate(3, 4, size=(3, 2))

    def test_randint_empty1(self):
        self.generate(3, 10, size=0)

    def test_randint_empty2(self):
        self.generate(3, size=(4, 0, 5))

    def test_randint_overflow(self):
        self.generate(numpy.int8(-100), numpy.int8(100))

    def test_randint_float1(self):
        self.generate(-1.2, 3.4, 5)

    def test_randint_float2(self):
        self.generate(6.7, size=(2, 3))

    def test_randint_int64_1(self):
        self.generate(2**34, 2**40, 3)


@testing.gpu
@testing.fix_random()
class TestUniform(RandomGeneratorTestCase):

    target_method = 'uniform'

    def test_uniform_1(self):
        self.generate()

    def test_uniform_2(self):
        self.generate(-4.2, 2.4, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_uniform_ks_1(self, dtype):
        self.check_ks(0.05)(
            size=2000, dtype=dtype)

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_uniform_ks_2(self, dtype):
        self.check_ks(0.05)(
            -4.2, 2.4, size=2000, dtype=dtype)


@testing.parameterize(
    {'mu': 0.0, 'kappa': 1.0},
    {'mu': 3.0, 'kappa': 3.0},
    {'mu': 3.0, 'kappa': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestVonmises(RandomGeneratorTestCase):

    target_method = 'vonmises'

    def test_vonmises(self):
        self.generate(mu=self.mu, kappa=self.kappa, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_vonmises_ks(self, dtype):
        self.check_ks(0.05)(
            self.mu, self.kappa, size=2000, dtype=dtype)


@testing.parameterize(
    {'mean': 1.0, 'scale': 3.0},
    {'mean': 3.0, 'scale': 3.0},
    {'mean': 3.0, 'scale': 1.0},
)
@testing.gpu
@testing.fix_random()
class TestWald(RandomGeneratorTestCase):

    target_method = 'wald'

    def test_wald(self):
        self.generate(mean=self.mean, scale=self.scale, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_wald_ks(self, dtype):
        self.check_ks(0.05)(
            self.mean, self.scale, size=2000, dtype=dtype)


@testing.parameterize(
    {'a': 0.5},
    {'a': 1.0},
    {'a': 3.0},
    {'a': numpy.inf},
)
@testing.gpu
@testing.fix_random()
class TestWeibull(RandomGeneratorTestCase):

    target_method = 'weibull'

    def test_weibull(self):
        self.generate(a=self.a, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_weibull_ks(self, dtype):
        self.check_ks(0.05)(
            a=self.a, size=2000, dtype=dtype)


@testing.parameterize(
    {'a': 2.0},
)
@testing.gpu
@testing.fix_random()
class TestZipf(RandomGeneratorTestCase):

    target_method = 'zipf'

    def test_zipf(self):
        self.generate(a=self.a, size=(3, 2))

    # TODO(kataoka): add distribution test


@testing.parameterize(
    {'a': 3, 'size': 5},
    {'a': [1, 2, 3], 'size': 5},
)
@testing.fix_random()
@testing.gpu
class TestChoiceReplaceFalseFailure(unittest.TestCase):

    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_choice_invalid_value(self, xp):
        rs = xp.random.RandomState(seed=testing.generate_seed())
        rs.choice(a=self.a, size=self.size, replace=False)


class TestResetStates(unittest.TestCase):

    def test_reset_states(self):
        generator._random_states = 'dummy'
        generator.reset_states()
        self.assertEqual({}, generator._random_states)


@testing.gpu
class TestGetRandomState(unittest.TestCase):

    def setUp(self):
        self.device_id = cuda.Device().id
        self.rs_tmp = generator._random_states

    def tearDown(self, *args):
        generator._random_states = self.rs_tmp

    def test_get_random_state_initialize(self):
        generator._random_states = {}
        rs = generator.get_random_state()
        self.assertEqual(generator._random_states[self.device_id], rs)

    def test_get_random_state_memoized(self):
        generator._random_states = {self.device_id: 'expected',
                                    self.device_id + 1: 'dummy'}
        rs = generator.get_random_state()
        self.assertEqual('expected', generator._random_states[self.device_id])
        self.assertEqual('dummy', generator._random_states[self.device_id + 1])
        self.assertEqual('expected', rs)


@testing.gpu
class TestSetRandomState(unittest.TestCase):

    def setUp(self):
        self.rs_tmp = generator._random_states

    def tearDown(self, *args):
        generator._random_states = self.rs_tmp

    def test_set_random_state(self):
        rs = generator.RandomState()
        generator.set_random_state(rs)
        assert generator.get_random_state() is rs

    def test_set_random_state_call_multiple_times(self):
        generator.set_random_state(generator.RandomState())
        rs = generator.RandomState()
        generator.set_random_state(rs)
        assert generator.get_random_state() is rs


@testing.gpu
@testing.fix_random()
class TestStandardExponential(RandomGeneratorTestCase):

    target_method = 'standard_exponential'

    def test_standard_exponential(self):
        self.generate(size=(3, 2))

    @attr.slow
    @condition.repeat(10)
    def test_standard_exponential_isfinite(self):
        x = self.generate(size=10**7)
        self.assertTrue(cupy.isfinite(x).all())

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_standard_exponential_ks(self, dtype):
        self.check_ks(0.05)(
            size=2000, dtype=dtype)


@testing.parameterize(
    {'left': -1.0, 'mode': 0.0, 'right': 2.0},
)
@testing.gpu
@testing.fix_random()
class TestTriangular(RandomGeneratorTestCase):

    target_method = 'triangular'

    def test_triangular(self):
        self.generate(
            left=self.left, mode=self.mode, right=self.right, size=(3, 2))


@testing.gpu
class TestRandomStateThreadSafe(unittest.TestCase):

    def setUp(self):
        cupy.random.reset_states()

    def test_get_random_state_thread_safe(self):
        seed = 10
        threads = [
            threading.Thread(target=lambda: cupy.random.seed(seed)),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
            threading.Thread(target=lambda: cupy.random.get_random_state()),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        actual = cupy.random.uniform()
        cupy.random.seed(seed)
        expected = cupy.random.uniform()
        self.assertEqual(actual, expected)

    def test_set_random_state_thread_safe(self):
        rs = cupy.random.RandomState()
        threads = [
            threading.Thread(target=lambda: cupy.random.set_random_state(rs)),
            threading.Thread(target=lambda: cupy.random.set_random_state(rs)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cupy.random.get_random_state() is rs


@testing.gpu
class TestGetRandomState2(unittest.TestCase):

    def setUp(self):
        self.rs_dict = generator._random_states
        generator._random_states = {}
        self.cupy_seed = os.getenv('CUPY_SEED')
        self.chainer_seed = os.getenv('CHAINER_SEED')

    def tearDown(self, *args):
        generator._random_states = self.rs_dict
        if self.cupy_seed is None:
            os.environ.pop('CUPY_SEED', None)
        else:
            os.environ['CUPY_SEED'] = self.cupy_seed
        if self.chainer_seed is None:
            os.environ.pop('CHAINER_SEED', None)
        else:
            os.environ['CHAINER_SEED'] = self.chainer_seed

    def test_get_random_state_no_cupy_no_chainer_seed(self):
        os.environ.pop('CUPY_SEED', None)
        os.environ.pop('CHAINER_SEED', None)
        rvs0 = self._get_rvs_reset()
        rvs1 = self._get_rvs_reset()

        self._check_different(rvs0, rvs1)

    def test_get_random_state_no_cupy_with_chainer_seed(self):
        rvs0 = self._get_rvs(generator.RandomState(5))

        os.environ.pop('CUPY_SEED', None)
        os.environ['CHAINER_SEED'] = '5'
        rvs1 = self._get_rvs_reset()

        self._check_same(rvs0, rvs1)

    def test_get_random_state_with_cupy_no_chainer_seed(self):
        rvs0 = self._get_rvs(generator.RandomState(6))

        os.environ['CUPY_SEED'] = '6'
        os.environ.pop('CHAINER_SEED', None)
        rvs1 = self._get_rvs_reset()

        self._check_same(rvs0, rvs1)

    def test_get_random_state_with_cupy_with_chainer_seed(self):
        rvs0 = self._get_rvs(generator.RandomState(7))

        os.environ['CUPY_SEED'] = '7'
        os.environ['CHAINER_SEED'] = '8'
        rvs1 = self._get_rvs_reset()

        self._check_same(rvs0, rvs1)

    def _get_rvs(self, rs):
        rvu = rs.rand(4)
        rvn = rs.randn(4)
        return rvu, rvn

    def _get_rvs_reset(self):
        generator.reset_states()
        return self._get_rvs(generator.get_random_state())

    def _check_same(self, rvs0, rvs1):
        for rv0, rv1 in zip(rvs0, rvs1):
            testing.assert_array_equal(rv0, rv1)

    def _check_different(self, rvs0, rvs1):
        for rv0, rv1 in zip(rvs0, rvs1):
            for r0, r1 in zip(rv0, rv1):
                self.assertNotEqual(r0, r1)


class TestCheckAndGetDtype(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def test_float32_64_type(self, dtype):
        self.assertEqual(generator._check_and_get_dtype(dtype),
                         numpy.dtype(dtype))

    def test_float16(self):
        with self.assertRaises(TypeError):
            generator._check_and_get_dtype(numpy.float16)

    @testing.for_int_dtypes()
    def test_int_type(self, dtype):
        with self.assertRaises(TypeError):
            generator._check_and_get_dtype(dtype)
