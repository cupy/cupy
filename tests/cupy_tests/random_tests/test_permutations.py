import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing import _condition


@testing.parameterize(
    {'seed': None},
    {'seed': 0},
)
@testing.gpu
class TestPermutations(unittest.TestCase):

    def _xp_random(self, xp):
        if self.seed is None:
            return xp.random
        else:
            return xp.random.RandomState(seed=self.seed)

    # Test ranks

    # TODO(niboshi): Fix xfail
    @pytest.mark.xfail(reason='Explicit error types required')
    def test_permutation_zero_dim(self):
        for xp in (numpy, cupy):
            xp_random = self._xp_random(xp)
            a = testing.shaped_random((), xp)
            with pytest.raises(IndexError):
                xp_random.permutation(a)

    # Test same values

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_permutation_sort_1dim(self, dtype):
        cupy_random = self._xp_random(cupy)
        a = cupy.arange(10, dtype=dtype)
        b = cupy.copy(a)
        c = cupy_random.permutation(a)
        testing.assert_allclose(a, b)
        testing.assert_allclose(b, cupy.sort(c))

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_permutation_sort_ndim(self, dtype):
        cupy_random = self._xp_random(cupy)
        a = cupy.arange(15, dtype=dtype).reshape(5, 3)
        b = cupy.copy(a)
        c = cupy_random.permutation(a)
        testing.assert_allclose(a, b)
        testing.assert_allclose(b, cupy.sort(c, axis=0))

    # Test seed

    @testing.for_all_dtypes()
    def test_permutation_seed1(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        b = cupy.copy(a)

        cupy_random = self._xp_random(cupy)
        if self.seed is None:
            cupy_random.seed(0)
        pa = cupy_random.permutation(a)
        cupy_random = self._xp_random(cupy)
        if self.seed is None:
            cupy_random.seed(0)
        pb = cupy_random.permutation(b)

        testing.assert_allclose(pa, pb)


@testing.gpu
class TestShuffle(unittest.TestCase):

    # Test ranks

    def test_shuffle_zero_dim(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp)
            with pytest.raises(TypeError):
                xp.random.shuffle(a)

    # Test same values

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_shuffle_sort_1dim(self, dtype):
        a = cupy.arange(10, dtype=dtype)
        b = cupy.copy(a)
        cupy.random.shuffle(a)
        testing.assert_allclose(cupy.sort(a), b)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_shuffle_sort_ndim(self, dtype):
        a = cupy.arange(15, dtype=dtype).reshape(5, 3)
        b = cupy.copy(a)
        cupy.random.shuffle(a)
        testing.assert_allclose(cupy.sort(a, axis=0), b)

    # Test seed

    @testing.for_all_dtypes()
    def test_shuffle_seed1(self, dtype):
        a = testing.shaped_random((10,), cupy, dtype)
        b = cupy.copy(a)
        cupy.random.seed(0)
        cupy.random.shuffle(a)
        cupy.random.seed(0)
        cupy.random.shuffle(b)
        testing.assert_allclose(a, b)


@testing.parameterize(*(testing.product({
    'num': [0, 1, 100, 1000, 10000, 100000],
})))
@testing.gpu
class TestPermutationSoundness(unittest.TestCase):

    def setUp(self):
        a = cupy.random.permutation(self.num)
        self.a = a.get()

    # Test soundness

    @_condition.repeat(3)
    def test_permutation_soundness(self):
        assert(numpy.sort(self.a) == numpy.arange(self.num)).all()


@testing.parameterize(*(testing.product({
    'offset': [0, 17, 34, 51],
    'gap': [1, 2, 3, 5, 7],
    'mask': [1, 2, 4, 8, 16, 32, 64, 128],
})))
@testing.gpu
class TestPermutationRandomness(unittest.TestCase):

    num = 256

    def setUp(self):
        a = cupy.random.permutation(self.num)
        self.a = a.get()
        self.num_half = int(self.num / 2)

    # Simple bit proportion test

    # This test is to check kind of randomness of permutation.
    # An intuition behind this test is that, when you make a sub-array
    # by regularly extracting half elements from the permuted array,
    # the sub-array should also hold randomeness and accordingly
    # frequency of appearance of 0 and 1 at each bit position of
    # whole elements in the sub-array should become similar
    # when elements count of original array is 2^N.
    # Note that this is not an establishd method to check randomness.
    # TODO(anaruse): implement randomness check using some established methods.
    @_condition.repeat_with_success_at_least(5, 3)
    def test_permutation_randomness(self):
        if self.mask > self.num_half:
            return
        index = numpy.arange(self.num_half)
        index = (index * self.gap + self.offset) % self.num
        samples = self.a[index]
        ret = (samples & self.mask > 0)
        count = numpy.count_nonzero(ret)  # expectation: self.num_half / 2
        if count > self.num_half - count:
            count = self.num_half - count
        prob_le_count = self._calc_probability(count)
        if prob_le_count < 0.001:
            raise

    def _calc_probability(self, count):
        comb_all = self._comb(self.num, self.num_half)
        comb_le_count = 0
        for i in range(count + 1):
            tmp = self._comb(self.num_half, i)
            comb_i = tmp * tmp
            comb_le_count += comb_i
        prob = comb_le_count / comb_all
        return prob

    def _comb(self, N, k):
        val = numpy.float64(1)
        for i in range(k):
            val *= (N - i) / (k - i)
        return val
