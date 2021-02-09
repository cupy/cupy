import functools
import threading
import unittest
import pytest

import numpy

import cupy
from cupy import random
from cupy import testing
from cupy.testing import attr
from cupy.testing import condition


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

            assert cupy_result is not None
            assert numpy_result is not None
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


@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class GeneratorTestCase(unittest.TestCase):

    target_method = None

    def setUp(self):
        self.__seed = testing.generate_seed()
        # TODO(ecastill) test it with other generators?
        self.rs = random._generator_api.Generator(
            random._bit_generator.Philox4x3210(seed=self.__seed))

    def _get_generator_func(self, *args, **kwargs):
        assert isinstance(self.target_method, str), (
            'generate_method must be overridden')
        f = getattr(self.rs, self.target_method)
        return lambda: f(*args, **kwargs)

    def _generate_check_repro(self, func, seed):
        # Sample a random array while checking reproducibility
        self.rs.bit_generator = random._bit_generator.Philox4x3210(seed=seed)
        x = func()
        self.rs.bit_generator = random._bit_generator.Philox4x3210(seed=seed)
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
        numpy_rs = numpy.random.Generator(numpy.random.MT19937(self.__seed))
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
    # Unfortunately cupy and numpy does not share the random generators
    if xp == cupy:
        bit_gen = cupy.random.Philox4x3210()
    else:
        bit_gen = numpy.random.MT19937()
    method = getattr(xp.random.Generator(bit_gen), method_name)
    if xp == cupy:
        return method

    def f(*args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        ret = method(*args, **kwargs)
        if dtype is not None:
            ret = ret.astype(dtype, copy=False)
        return ret

    return f


@testing.parameterize(
    {'a': 1.0, 'b': 3.0},
    {'a': 3.0, 'b': 3.0},
    {'a': 3.0, 'b': 1.0},
)
@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestBeta(GeneratorTestCase):

    target_method = 'beta'

    def test_beta(self):
        self.generate(a=self.a, b=self.b, size=(3, 2))

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_beta_ks(self, dtype):
        self.check_ks(0.05)(
            a=self.a, b=self.b, size=2000, dtype=dtype)


@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestStandardExponential(GeneratorTestCase):

    target_method = 'standard_exponential'

    def test_standard_exponential(self):
        self.generate(size=(3, 2))

    @attr.slow
    @condition.repeat(10)
    def test_standard_exponential_isfinite(self):
        x = self.generate(size=10**7)
        assert cupy.isfinite(x).all()

    @testing.for_dtypes('fd')
    @condition.repeat_with_success_at_least(10, 3)
    def test_standard_exponential_ks(self, dtype):
        self.check_ks(0.05)(size=2000, dtype=dtype)


@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestIntegers(GeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'integers'

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


@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestRandomStateThreadSafe(unittest.TestCase):

    def test_default_rng_thread_safe(self):
        seed = 10
        threads = [
            threading.Thread(target=lambda: cupy.random.default_rng(seed)),
            threading.Thread(target=lambda: cupy.random.default_rng()),
            threading.Thread(target=lambda: cupy.random.default_rng()),
            threading.Thread(target=lambda: cupy.random.default_rng()),
            threading.Thread(target=lambda: cupy.random.default_rng()),
            threading.Thread(target=lambda: cupy.random.default_rng()),
            threading.Thread(target=lambda: cupy.random.default_rng()),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        actual = cupy.random.default_rng(seed).standard_exponential()
        expected = cupy.random.default_rng(seed).standard_exponential()
        assert actual == expected
