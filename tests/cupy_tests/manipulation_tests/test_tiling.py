from __future__ import annotations

import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 1},
    {'repeats': 2, 'axis': -1},
    {'repeats': [0, 0, 0], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': 1},
    {'repeats': [1, 2, 3], 'axis': -2},
)
class TestRepeat(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 1},
)
class TestRepeatListBroadcast:

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': 0, 'axis': None},
    {'repeats': 2, 'axis': None},
    {'repeats': 2, 'axis': 0},
    {'repeats': [1, 2, 3, 4], 'axis': None},
    {'repeats': [1, 2, 3, 4], 'axis': 0},
)
class TestRepeat1D:

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': [2], 'axis': None},
    {'repeats': [2], 'axis': 0},
)
class TestRepeat1DListBroadcast:

    @testing.numpy_cupy_array_equal()
    def test_array_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    {'repeats': -3, 'axis': None},
    {'repeats': [-3, -3], 'axis': 0},
    {'repeats': [1, 2, 3], 'axis': None},
    {'repeats': [1, 2], 'axis': 1},
    {'repeats': 2, 'axis': -4},
    {'repeats': 2, 'axis': 3},
)
class TestRepeatFailure:

    def test_repeat_failure(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.repeat(x, self.repeats, self.axis)


@testing.parameterize(
    # 1-D
    {'shape': (6,), 'reps': [1, 3, 2, 1, 1, 2], 'axis': None},
    {'shape': (6,), 'reps': [2],                 'axis': None},
    # 2-D
    {'shape': (2, 3), 'reps': [2, 1],     'axis': 0},
    {'shape': (2, 3), 'reps': [1, 3, 2],  'axis': 1},
    {'shape': (2, 3), 'reps': [2],        'axis': 0},
    {'shape': (2, 3), 'reps': [2],        'axis': 1},
    # 3-D
    {'shape': (2, 3, 4), 'reps': [1, 2, 3, 4], 'axis': 2},
    {'shape': (2, 3, 4), 'reps': [0, 3],        'axis': 0},
    {'shape': (2, 3, 4), 'reps': [1, 2, 3],     'axis': 1},
    {'shape': (2, 3, 4), 'reps': [4],            'axis': 2},
    # negative axis
    {'shape': (2, 3, 4), 'reps': [1, 2, 3, 4], 'axis': -1},
    {'shape': (2, 3, 4), 'reps': [1, 2, 3],    'axis': -2},
    # axis=None
    {'shape': (2, 3), 'reps': [1, 2, 3, 4, 5, 0], 'axis': None},
    {'shape': (4,),   'reps': [0, 0, 0, 0],        'axis': None},
    {'shape': (4,),   'reps': [5, 0, 3, 1],        'axis': None},
    # zeros in reps
    {'shape': (4,),   'reps': [0, 2, 0, 1], 'axis': 0},
    {'shape': (2, 3), 'reps': [0, 3, 0],    'axis': 1},
    # broadcast
    {'shape': (2, 3), 'reps': [0], 'axis': 0},
    {'shape': (2, 3), 'reps': [1], 'axis': 1},
    {'shape': (3, 4), 'reps': [2], 'axis': None},
    # 4-D
    {'shape': (2, 3, 4, 5), 'reps': [2, 1, 3], 'axis': 1},
    # empty
    {'shape': (0, 3), 'reps': [2],       'axis': 0},
    {'shape': (2, 3), 'reps': [0, 0, 0], 'axis': 1},
)
class TestRepeatNdarrayRepeats:
    """ndarray repeats matches numpy for diverse shapes, axes, and reps."""

    @testing.numpy_cupy_array_equal()
    def test_repeat(self, xp):
        x = testing.shaped_arange(self.shape, xp)
        return xp.repeat(x, xp.array(self.reps), self.axis)


@testing.parameterize(*[{'rep_dtype': d} for d in [
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32,
]])
class TestRepeatNdarrayRepsDtype:
    """Various integer dtypes for reps are accepted."""

    @testing.numpy_cupy_array_equal()
    def test_repeat(self, xp):
        x = testing.shaped_arange((4,), xp)
        return xp.repeat(x, xp.array([1, 2, 3, 4], dtype=self.rep_dtype), 0)


@testing.parameterize(*[{'a_dtype': d} for d in [
    numpy.bool_, numpy.int32, numpy.float32, numpy.float64,
    numpy.complex64,
]])
class TestRepeatNdarrayArrayDtype:
    """Output dtype matches input dtype."""

    @testing.numpy_cupy_array_equal()
    def test_dtype_preserved(self, xp):
        x = testing.shaped_arange((3, 4), xp, dtype=self.a_dtype)
        return xp.repeat(x, xp.array([1, 2, 3, 4]), axis=1)


class TestRepeatNdarrayNonContiguous:

    @testing.numpy_cupy_array_equal()
    def test_transposed(self, xp):
        x = testing.shaped_arange((4, 3), xp).T
        return xp.repeat(x, xp.array([2, 1, 3, 0]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_strided(self, xp):
        x = testing.shaped_arange((3, 8), xp)[:, ::2]
        return xp.repeat(x, xp.array([1, 2, 3, 0]), axis=1)

    @testing.numpy_cupy_array_equal()
    def test_reversed(self, xp):
        x = testing.shaped_arange((5,), xp)[::-1]
        return xp.repeat(x, xp.array([0, 1, 2, 1, 0]))


class TestRepeatScalarEquivalence:
    """All scalar-like repeats inputs produce identical results."""

    def _check_all_equal(self, a, n, axis):
        expected = cupy.array(numpy.repeat(cupy.asnumpy(a), n, axis))
        for form in [n, numpy.intp(n), [n],
                     cupy.array([n]), cupy.array(n)]:
            cupy.testing.assert_array_equal(
                cupy.repeat(a, form, axis), expected)

    def test_equivalence(self):
        a = cupy.arange(6).reshape(2, 3)
        for n, axis in [(3, None), (2, 0), (4, 1), (0, 0), (1, 0)]:
            self._check_all_equal(a, n, axis)

    def test_negative_raises(self):
        a = cupy.arange(3)
        for form in [-1, numpy.intp(-1), [-1],
                     cupy.array([-1]), cupy.array(-1)]:
            with pytest.raises(ValueError, match=r'negative'):
                cupy.repeat(a, form)

    def test_numpy_scalar_accepted(self):
        a = cupy.arange(3)
        cupy.testing.assert_array_equal(
            cupy.repeat(a, numpy.int64(2)), cupy.repeat(a, 2))

    def test_numpy_ndarray_rejected(self):
        a = cupy.arange(3)
        with pytest.raises(TypeError, match='numpy.ndarray'):
            cupy.repeat(a, numpy.array([1, 2, 3]))


class TestRepeatNdarrayErrors:

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match=r'same length'):
            cupy.repeat(cupy.arange(4), cupy.array([1, 2]), axis=0)

    def test_negative(self):
        with pytest.raises(ValueError, match=r'negative'):
            cupy.repeat(cupy.arange(3), cupy.array([-1, 1, 2]))

    def test_float_dtype_matches_numpy(self):
        # Both NumPy and CuPy raise TypeError for unsafe cast
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.repeat(xp.arange(3), xp.array([1.0, 1.0, 1.0]), 0)

    def test_uint64_matches_numpy(self):
        # Both reject uint64 (unsigned → signed is unsafe)
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.repeat(xp.arange(3),
                          xp.array([1, 2, 3], dtype=numpy.uint64))

    def test_ndim_gt1_matches_numpy(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.repeat(xp.arange(6), xp.array([[1, 2, 3, 4, 5, 6]]))

    def test_ndim_gt1_list_rejected(self):
        with pytest.raises(ValueError, match=r'too deep'):
            cupy.repeat(cupy.arange(6), [[1, 2, 3, 4, 5, 6]])

    def test_bad_axis(self):
        with pytest.raises(Exception):
            cupy.repeat(cupy.arange(12).reshape(3, 4),
                        cupy.array([1, 2, 3]), axis=5)

    def test_method_interface(self):
        a = cupy.arange(4)
        reps = cupy.array([1, 2, 0, 3])
        cupy.testing.assert_array_equal(
            a.repeat(reps), cupy.repeat(a, reps))


class TestRepeatNdarrayDtypeEdges:

    @testing.numpy_cupy_array_equal()
    def test_bool_perelement(self, xp):
        return xp.repeat(xp.arange(3), xp.array([True, False, True]))

    @testing.numpy_cupy_array_equal()
    def test_bool_broadcast(self, xp):
        return xp.repeat(testing.shaped_arange((3, 4), xp),
                         xp.array([True]), axis=0)

    @testing.numpy_cupy_array_equal()
    def test_uint32_accepted(self, xp):
        return xp.repeat(xp.arange(4),
                         xp.array([1, 2, 3, 4], dtype=numpy.uint32))


class TestRepeatNdarrayLarge:

    @testing.numpy_cupy_array_equal()
    def test_large_single(self, xp):
        return xp.repeat(testing.shaped_arange((3,), xp),
                         xp.array([0, 100000, 0]))

    @testing.numpy_cupy_array_equal()
    def test_large_broadcast(self, xp):
        return xp.repeat(testing.shaped_arange((3,), xp),
                         xp.array([50000]))


@testing.parameterize(
    {'reps': 0},
    {'reps': 1},
    {'reps': 2},
    {'reps': (0, 1)},
    {'reps': (2, 3)},
    {'reps': (2, 3, 4, 5)},
)
class TestTile(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_tile(self, xp):
        x = testing.shaped_arange((2, 3, 4), xp)
        return xp.tile(x, self.reps)


@testing.parameterize(
    {'reps': -1},
    {'reps': (-1, -2)},
)
class TestTileFailure(unittest.TestCase):

    def test_tile_failure(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(ValueError):
                xp.tile(x, -3)
