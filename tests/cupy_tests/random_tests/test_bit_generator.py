import unittest
import pytest

import numpy

import cupy
from cupy import random
from cupy import testing


class BitGeneratorTestCase:

    def setUp(self):
        self.seed = testing.generate_seed()

    def check_seed(self, seed):
        bg1 = self.bg(seed)
        bg2 = self.bg(seed)
        bg3 = self.bg(None)

        xs1 = bg1.random_raw(10)
        xs2 = bg2.random_raw(10)
        xs3 = bg3.random_raw(10)

        # Random state must be reproducible
        assert cupy.array_equal(xs1, xs2)
        # Random state must be initialized randomly with seed=None
        assert not cupy.array_equal(xs1, xs3)

    @testing.for_int_dtypes(no_bool=True)
    def test_seed_not_none(self, dtype):
        self.check_seed(dtype(0))

    @testing.for_dtypes([numpy.complex_])
    def test_seed_invalid_type_complex(self, dtype):
        with self.assertRaises(TypeError):
            self.bg(dtype(0))

    @testing.for_float_dtypes()
    def test_seed_invalid_type_float(self, dtype):
        with self.assertRaises(TypeError):
            self.bg(dtype(0))

    def test_array_seed(self):
        self.check_seed(numpy.random.randint(0, 2**31, size=10))


@testing.with_requires('numpy>=1.17.0')
@testing.fix_random()
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestBitGeneratorXORWOW(BitGeneratorTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bg = random._bit_generator.XORWOW


@testing.with_requires('numpy>=1.17.0')
@testing.fix_random()
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestBitGeneratorMRG32k3a(BitGeneratorTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bg = random._bit_generator.MRG32k3a


@testing.with_requires('numpy>=1.17.0')
@testing.fix_random()
@testing.gpu
@pytest.mark.skipif(cupy.cuda.runtime.is_hip,
                    reason='HIP does not support this')
class TestBitGeneratorPhilox4x3210(BitGeneratorTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bg = random._bit_generator.Philox4x3210
