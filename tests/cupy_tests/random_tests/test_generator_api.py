import threading
import unittest
import pytest

import numpy

import cupy
from cupy import random
from cupy import testing
from cupy.testing import _condition

from cupy_tests.random_tests import common_distributions


@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class GeneratorTestCase(common_distributions.BaseGeneratorTestCase):

    target_method = None

    def get_rng(self, xp, seed):
        if xp is cupy:
            return cupy.random._generator_api.Generator(
                random._bit_generator.Philox4x3210(seed=seed))
        else:
            return numpy.random.Generator(numpy.random.MT19937(seed))

    def set_rng_seed(self, seed):
        self.rng.bit_generator = random._bit_generator.Philox4x3210(seed=seed)


class InvalidOutsMixin:

    def invalid_dtype_out(self, **kwargs):
        out = cupy.zeros((3, 2), dtype=cupy.float32)
        with pytest.raises(TypeError):
            self.generate(size=(3, 2), out=out, **kwargs)

    def invalid_contiguity(self, **kwargs):
        out = cupy.zeros((4, 6), dtype=cupy.float64)[0:3:, 0:2:]
        with pytest.raises(ValueError):
            self.generate(size=(3, 2), out=out, **kwargs)

    def invalid_shape(self, **kwargs):
        out = cupy.zeros((3, 3), dtype=cupy.float64)
        with pytest.raises(ValueError):
            self.generate(size=(3, 2), out=out, **kwargs)

    def test_invalid_dtype_out(self):
        self.invalid_dtype_out()

    def test_invalid_contiguity(self):
        self.invalid_contiguity()

    def test_invalid_shape(self):
        self.invalid_shape()


@testing.parameterize(*common_distributions.exponential_params)
@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestExponential(
    common_distributions.Exponential,
    GeneratorTestCase
):
    pass


@testing.parameterize(*common_distributions.poisson_params)
@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestPoisson(
    common_distributions.Poisson,
    GeneratorTestCase
):
    pass


@testing.parameterize(*common_distributions.beta_params)
@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestBeta(
    common_distributions.Beta,
    GeneratorTestCase
):
    pass


@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestStandardExponential(
    InvalidOutsMixin,
    common_distributions.StandardExponential,
    GeneratorTestCase,
):
    pass


@testing.parameterize(*common_distributions.gamma_params)
@testing.gpu
@testing.fix_random()
class TestGamma(
    common_distributions.Gamma,
    GeneratorTestCase,
):
    pass


@testing.parameterize(*common_distributions.standard_gamma_params)
@testing.gpu
@testing.fix_random()
class TestStandardGamma(
    common_distributions.StandardGamma,
    GeneratorTestCase,
):
    pass


@testing.gpu
@testing.fix_random()
class TestStandardGammaInvalid(InvalidOutsMixin, GeneratorTestCase):

    target_method = 'standard_gamma'

    def test_invalid_dtype_out(self):
        self.invalid_dtype_out(shape=1.0)

    def test_invalid_contiguity(self):
        self.invalid_contiguity(shape=1.0)

        out = cupy.zeros((4, 6), order='F', dtype=cupy.float64)
        with pytest.raises(ValueError):
            self.generate(size=(4, 6), out=out, shape=1.0)

    def test_invalid_shape(self):
        self.invalid_shape(shape=1.0)

    def test_invalid_dtypes(self):
        for dtype in 'bhiqleFD':
            with pytest.raises(TypeError):
                self.generate(size=(3, 2), shape=1.0, dtype=dtype)


@testing.gpu
@testing.fix_random()
class TestStandardGammaEmpty(GeneratorTestCase):

    target_method = 'standard_gamma'

    def test_empty_shape(self):
        y = self.generate(shape=cupy.empty((1, 0)))
        assert y.shape == (1, 0)

    def test_empty_size(self):
        y = self.generate(1.0, size=(1, 0))
        assert y.shape == (1, 0)

    def test_empty_out(self):
        out = cupy.empty((1, 0))
        y = self.generate(cupy.empty((1, 0)), out=out)
        assert y is out
        assert y.shape == (1, 0)


@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.parameterize(*common_distributions.standard_normal_params)
@testing.fix_random()
class TestStandardNormal(
    common_distributions.StandardNormal,
    GeneratorTestCase
):
    pass


@testing.with_requires('numpy>=1.17.0')
@testing.gpu
@testing.fix_random()
class TestStandardNormalInvalid(InvalidOutsMixin, GeneratorTestCase):

    target_method = 'standard_normal'

    def test_invalid_dtypes(self):
        for dtype in 'bhiqleFD':
            with pytest.raises(TypeError):
                self.generate(size=(3, 2), dtype=dtype)


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
@testing.fix_random()
class TestRandom(InvalidOutsMixin, GeneratorTestCase):
    # TODO(niboshi):
    #   Test soundness of distribution.
    #   Currently only reprocibility is checked.

    target_method = 'random'

    def test_random(self):
        self.generate(3)

    @testing.for_dtypes('fd')
    @_condition.repeat_with_success_at_least(10, 3)
    def test_random_ks(self, dtype):
        self.check_ks(0.05)(size=2000, dtype=dtype)


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
