from __future__ import annotations

import unittest

import numpy
import pytest

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

    @testing.for_dtypes([numpy.complex128])
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
class TestBitGeneratorXORWOW(BitGeneratorTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bg = random._bit_generator.XORWOW


@testing.with_requires('numpy>=1.17.0')
@testing.fix_random()
class TestBitGeneratorMRG32k3a(BitGeneratorTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bg = random._bit_generator.MRG32k3a


@testing.with_requires('numpy>=1.17.0')
@testing.fix_random()
class TestBitGeneratorPhilox4x3210(BitGeneratorTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bg = random._bit_generator.Philox4x3210


@pytest.mark.parametrize('bit_generator',
                         [random.XORWOW, random.MRG32k3a, random.Philox4x3210])
@pytest.mark.parametrize('state_size', [1, 255, 256, 257])
@pytest.mark.parametrize('method', ['random_raw', 'random', 'standard_normal',
                                    'binomial'])
def test_partial_state_block(bit_generator, state_size, method):
    aligned_size = ((state_size + 255) // 256) * 256
    reference = bit_generator(42, size=aligned_size)

    def padded_allocator(size):
        # Keep any access by padded threads inside the allocation, so a
        # regression fails without corrupting unrelated device memory.
        padded_size = size // state_size * aligned_size
        mem = cupy.cuda.MemoryPointer(cupy.cuda.Memory(padded_size), 0)
        mem.copy_from_device(reference._state.data, padded_size)
        return mem

    with cupy.cuda.using_allocator(padded_allocator):
        bg = bit_generator(42, size=state_size)
    state = bg._state
    padding = cupy.ndarray(
        (state.data.mem.size - state.nbytes,), dtype=numpy.int8,
        memptr=state.data + state.nbytes)
    padding_before = padding.copy()
    if method == 'random_raw':
        generate = bg.random_raw
        generate_reference = reference.random_raw
        args = ()
    else:
        generate = getattr(random.Generator(bg), method)
        generate_reference = getattr(random.Generator(reference), method)
        args = (20, 0.25) if method == 'binomial' else ()

    # A single call must advance each valid state just as three full rounds
    # do. Padded threads must neither generate samples nor update state.
    actual = generate(*args, size=3 * state_size)
    expected = cupy.concatenate([
        generate_reference(*args, size=aligned_size)[:state_size]
        for _ in range(3)])
    testing.assert_array_equal(actual, expected)
    testing.assert_array_equal(padding, padding_before)
