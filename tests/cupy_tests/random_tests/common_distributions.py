import functools
import unittest

import numpy

import cupy
from cupy import testing
from cupy.testing import _attr
from cupy.testing import _condition


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


class BaseGeneratorTestCase(unittest.TestCase):

    target_method = None

    def get_rng(self, xp, seed):
        pass

    def set_rng_seed(self, seed):
        pass

    def setUp(self):
        self.__seed = testing.generate_seed()
        # rng will be a new or old generator API object
        self.rng = self.get_rng(cupy, self.__seed)

    def _get_generator_func(self, *args, **kwargs):
        assert isinstance(self.target_method, str), (
            'generate_method must be overridden')
        f = getattr(self.rng, self.target_method)
        return lambda: f(*args, **kwargs)

    def _generate_check_repro(self, func, seed):
        # Sample a random array while checking reproducibility
        self.set_rng_seed(seed)
        x = func()
        self.set_rng_seed(seed)
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
        numpy_rng = self.get_rng(numpy, self.__seed)
        vals_numpy = getattr(numpy_rng, self.target_method)(*args, **kwargs)
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


beta_params = [
    {'a': 1.0, 'b': 3.0},
    {'a': 3.0, 'b': 3.0},
    {'a': 3.0, 'b': 1.0},
    {'a': [1.0, 3.0, 5.0, 6.0, 9.0], 'b':7.0},
    {'a': 5.0, 'b': [1.0, 5.0, 8.0, 1.0, 3.0]},
    {'a': [8.0, 6.0, 2.0, 4.0, 7.0], 'b':[3.0, 1.0, 2.0, 8.0, 1.0]}]


class Beta:

    target_method = 'beta'

    def test_beta(self):
        a = self.a
        b = self.b
        if (isinstance(self.a, list) or isinstance(self.b, list)):
            a = cupy.array(self.a)
            b = cupy.array(self.b)
        self.generate(a, b, size=(3, 5))

    @_condition.repeat_with_success_at_least(10, 3)
    def test_beta_ks(self):
        if (isinstance(self.a, list) or isinstance(self.b, list)):
            self.skipTest('Stastical checks only for scalar args')
        self.check_ks(0.05)(a=self.a, b=self.b, size=2000)


class StandardExponential:

    target_method = 'standard_exponential'

    def test_standard_exponential(self):
        self.generate(size=(3, 2))

    @_attr.slow
    @_condition.repeat(10)
    def test_standard_exponential_isfinite(self):
        x = self.generate(size=10**7)
        assert cupy.isfinite(x).all()

    @testing.for_dtypes('fd')
    @_condition.repeat_with_success_at_least(10, 3)
    def test_standard_exponential_ks(self, dtype):
        self.check_ks(0.05)(size=2000, dtype=dtype)


standard_gamma_params = [
    {'shape': 0.5},
    {'shape': 1.0},
    {'shape': 3.0}]


class StandardGamma:

    target_method = 'standard_gamma'

    def test_standard_gamma(self):
        self.generate(shape=self.shape, size=(3, 2))

    @testing.for_dtypes('fd')
    @_condition.repeat_with_success_at_least(10, 3)
    def test_standard_gamma_ks(self, dtype):
        self.check_ks(0.05)(
            shape=self.shape, size=2000, dtype=dtype)


standard_normal_params = [
    {'size': None},
    {'size': (1, 2, 3)},
    {'size': 3},
    {'size': (1000, 1000)},
    {'size': (3, 3)},
    {'size': ()}]


class StandardNormal:

    target_method = 'standard_normal'

    @testing.for_dtypes('fd')
    @_condition.repeat_with_success_at_least(10, 3)
    def test_normal_ks(self, dtype):
        self.check_ks(0.05)(size=self.size, dtype=dtype)


exponential_params = [
    {'scale': 0.5},
    {'scale': 1},
    {'scale': 10}]


class Exponential:

    target_method = 'exponential'

    def test_exponential(self):
        self.generate(scale=self.scale, size=(3, 2))

    @_condition.repeat_with_success_at_least(10, 3)
    def test_exponential_ks(self):
        self.check_ks(0.05)(
            self.scale, size=2000)


poisson_params = [
    {'lam': 1.0},
    {'lam': 3.0},
    {'lam': 10.0}]


class Poisson:

    target_method = 'poisson'

    def test_poisson(self):
        self.generate(lam=self.lam, size=(3, 2))

    @_condition.repeat_with_success_at_least(10, 3)
    def test_poisson_ks(self):
        self.check_ks(0.05)(
            lam=self.lam, size=2000)

    def test_poisson_large(self):
        self.generate(lam=self.lam, size=(1000, 1000))


gamma_params = [
    {'shape': 0.5, 'scale': 0.5},
    {'shape': 1.0, 'scale': 0.5},
    {'shape': 3.0, 'scale': 0.5},
    {'shape': 0.5, 'scale': 1.0},
    {'shape': 1.0, 'scale': 1.0},
    {'shape': 3.0, 'scale': 1.0},
    {'shape': 0.5, 'scale': 3.0},
    {'shape': 1.0, 'scale': 3.0},
    {'shape': 3.0, 'scale': 3.0}]


class Gamma:

    target_method = 'gamma'

    def test_gamma_1(self):
        self.generate(shape=self.shape, scale=self.scale, size=(3, 2))

    def test_gamma_2(self):
        self.generate(shape=self.shape, size=(3, 2))

    @_condition.repeat_with_success_at_least(10, 3)
    def test_gamma_ks(self):
        self.check_ks(0.05)(
            self.shape, self.scale, size=2000)


geometric_params = [
    {'p': 0.5},
    {'p': 0.1},
    {'p': 1.0},
    {'p': [0.1, 0.5]},
]


class Geometric:

    target_method = 'geometric'

    def test_geometric(self):
        p = self.p
        if not isinstance(self.p, float):
            p = cupy.array(self.p)
        self.generate(p=p, size=(3, 2))

    @_condition.repeat_with_success_at_least(10, 3)
    def test_geometric_ks(self):
        if not isinstance(self.p, float):
            self.skipTest('Statistical checks only for scalar `p`')
        self.check_ks(0.05)(
            p=self.p, size=2000)


hypergeometric_params = [
    {'ngood': 5, 'nbad': 5, 'nsample': 5},
    {'ngood': 10.0, 'nbad': 10.0, 'nsample': 10.0},
    {'ngood': 100.0, 'nbad': 2.0, 'nsample': 10.0},
    {'ngood': [0, 5, 8], 'nbad': [5, 0, 3], 'nsample': [2, 1, 8]},
    {'ngood': [1, 4, 2, 7, 6], 'nbad': 5.0, 'nsample': [2, 7, 4, 6, 5]},
]


class Hypergeometric:

    target_method = 'hypergeometric'

    def test_hypergeometric(self):
        ngood = self.ngood
        nbad = self.nbad
        nsample = self.nsample
        if (isinstance(self.ngood, list) or isinstance(self.nbad, list)
                or isinstance(self.nsample, list)):
            ngood = cupy.array(self.ngood)
            nbad = cupy.array(self.nbad)
            nsample = cupy.array(self.nsample)
        self.generate(ngood, nbad, nsample)

    @_condition.repeat_with_success_at_least(10, 3)
    def test_hypergeometric_ks(self):
        if (isinstance(self.ngood, list) or isinstance(self.nbad, list)
                or isinstance(self.nsample, list)):
            self.skipTest('Stastical checks only for scalar args')
        self.check_ks(0.05)(self.ngood, self.nbad, self.nsample, size=2000)


power_params = [
    {'a': 0.5},
    {'a': 1},
    {'a': 5},
    {'a': [0.8, 0.7, 1, 2, 5]},
]


class Power:

    target_method = 'power'

    def test_power(self):
        a = self.a
        if not isinstance(self.a, float):
            a = cupy.array(self.a)
        self.generate(a=a)

    @_condition.repeat_with_success_at_least(10, 3)
    def test_power_ks(self):
        if not isinstance(self.a, float):
            self.skipTest('Statistical checks only for scalar `a`')
        self.check_ks(0.05)(
            a=self.a, size=2000)


logseries_params = [
    {'p': 0.5},
    {'p': 0.1},
    {'p': 0.9},
    {'p': [0.8, 0.7]},
]


class Logseries:

    target_method = 'logseries'

    def test_logseries(self):
        p = self.p
        if not isinstance(self.p, float):
            p = cupy.array(self.p)
        self.generate(p=p, size=(3, 2))

    @_condition.repeat_with_success_at_least(10, 3)
    def test_geometric_ks(self):
        if not isinstance(self.p, float):
            self.skipTest('Statistical checks only for scalar `p`')
        self.check_ks(0.05)(p=self.p, size=2000)


chisquare_params = [
    {'df': 1.0},
    {'df': 3.0},
    {'df': 10.0},
    {'df': [2, 5, 8]},
]


class Chisquare:

    target_method = 'chisquare'

    def test_chisquare(self):
        df = self.df
        if not isinstance(self.df, float):
            df = cupy.array(self.df)
        self.generate(df=df)

    @_condition.repeat_with_success_at_least(10, 3)
    def test_chisquare_ks(self):
        if not isinstance(self.df, float):
            self.skipTest('Statistical checks only for scalar `df`')
        self.check_ks(0.05)(
            df=self.df, size=2000)


f_params = [
    {'dfnum': 1.0, 'dfden': 3.0},
    {'dfnum': 3.0, 'dfden': 3.0},
    {'dfnum': 3.0, 'dfden': 1.0},
    {'dfnum': [1.0, 3.0, 3.0], 'dfden': [3.0, 3.0, 1.0]},
]


class F:

    target_method = 'f'

    def test_f(self):
        dfnum = self.dfnum
        dfden = self.dfden
        if isinstance(self.dfnum, list) or isinstance(self.dfden, list):
            dfnum = cupy.array(self.dfnum)
            dfden = cupy.array(self.dfden)
        self.generate(dfnum, dfden)

    @_condition.repeat_with_success_at_least(10, 3)
    def test_f_ks(self):
        if isinstance(self.dfnum, list) or isinstance(self.dfden, list):
            self.skipTest('Stastical checks only for scalar args')
        self.check_ks(0.05)(self.dfnum, self.dfden, size=2000)


dirichlet_params = [
    {'alpha': 5},
    {'alpha': 1},
    {'alpha': [2, 5, 8]}
]


class Dirichlet:
    target_method = 'dirichlet'

    def test_dirichlet(self):
        alpha = self.alpha
        if not isinstance(self.alpha, float):
            alpha = cupy.array(self.alpha)
        self.generate(alpha=alpha, size=(3, 2))

    def test_dirichlet_int_shape(self):
        alpha = self.alpha
        if not isinstance(self.alpha, int):
            alpha = cupy.array(self.alpha)
        self.generate(alpha=alpha, size=5)

    # TODO(kataoka): add distribution test
