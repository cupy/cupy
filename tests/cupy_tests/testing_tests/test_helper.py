import unittest

import numpy
import pytest

import cupy
from cupy import testing


class TestPackageRequirements:
    def test_installed(self):
        assert testing.installed('cupy')
        assert testing.installed('cupy>9', 'numpy>=1.12')
        assert testing.installed('numpy>=1.10,<=2.0')
        assert not testing.installed('numpy>=2.0')
        assert not testing.installed('numpy>1.10,<1.9')

    def test_numpy_satisfies(self):
        assert testing.numpy_satisfies('>1.10')
        assert not testing.numpy_satisfies('>=2.10')

    @testing.with_requires('numpy>2.0')
    def test_with_requires(self):
        assert False, 'this should not happen'


@testing.parameterize(*testing.product({
    'xp': [numpy, cupy],
    'shape': [(3, 2), (), (3, 0, 2)],
}))
@testing.gpu
class TestShapedRandom(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_shape_and_dtype(self, dtype):
        a = testing.shaped_random(self.shape, self.xp, dtype)
        assert isinstance(a, self.xp.ndarray)
        assert a.shape == self.shape
        assert a.dtype == dtype

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_value_range(self, dtype):
        a = testing.shaped_random(self.shape, self.xp, dtype)
        assert self.xp.all(0 <= a)
        assert self.xp.all(a < 10)

    @testing.for_complex_dtypes()
    def test_complex(self, dtype):
        a = testing.shaped_random(self.shape, self.xp, dtype)
        assert self.xp.all(0 <= a.real)
        assert self.xp.all(a.real < 10)
        assert self.xp.all(0 <= a.imag)
        assert self.xp.all(a.imag < 10)
        if 0 not in self.shape:
            assert self.xp.any(a.imag)


@testing.parameterize(*testing.product({
    'xp': [numpy, cupy],
}))
@testing.gpu
class TestShapedRandomBool(unittest.TestCase):

    def test_bool(self):
        a = testing.shaped_random(10000, self.xp, numpy.bool_)
        assert 4000 < self.xp.sum(a) < 6000


@testing.parameterize(*testing.product({
    'dtype': [
        numpy.float16, numpy.float32, numpy.float64,
        numpy.complex64, numpy.complex128,
    ],
    'xp': [numpy, cupy],
    'x_s_shapes': [
        ((0, 0), (0,)),
        ((2, 2), (2,)),
        ((2, 3), (2,)),
        ((3, 2), (2,)),
        # broadcast
        ((2, 2), ()),
    ],
}))
class TestGenerateMatrix(unittest.TestCase):

    def test_generate_matrix(self):
        dtype = self.dtype
        x_shape, s_shape = self.x_s_shapes
        sv = self.xp.random.uniform(
            0.5, 1.5, s_shape).astype(dtype().real.dtype)
        x = testing.generate_matrix(
            x_shape, xp=self.xp, dtype=dtype, singular_values=sv)
        assert x.shape == x_shape

        if 0 in x_shape:
            return

        s = self.xp.linalg.svd(
            x.astype(numpy.complex128), full_matrices=False, compute_uv=False,
        )
        sv = self.xp.broadcast_to(sv, s.shape)
        sv_sorted = self.xp.sort(sv, axis=-1)[..., ::-1]

        rtol = 1e-3 if dtype == numpy.float16 else 1e-7
        self.xp.testing.assert_allclose(s, sv_sorted, rtol=rtol)


class TestGenerateMatrixInvalid(unittest.TestCase):

    def test_no_singular_values(self):
        with self.assertRaises(TypeError):
            testing.generate_matrix((2, 2))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix((2,), singular_values=1)

    def test_invalid_dtype_singular_values(self):
        with self.assertRaises(TypeError):
            testing.generate_matrix((2, 2), singular_values=1 + 0j)

    def test_invalid_dtype(self):
        with self.assertRaises(TypeError):
            testing.generate_matrix(
                (2, 2), dtype=numpy.int32, singular_values=1)

    def test_negative_singular_values(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix((2, 2), singular_values=[1, -1])

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix(
                (2, 2), singular_values=numpy.ones(3))

    def test_shape_mismatch_2(self):
        with self.assertRaises(ValueError):
            testing.generate_matrix(
                (0, 2, 2), singular_values=numpy.ones(3))


class TestAssertFunctionIsCalled(unittest.TestCase):

    def test_patch_ndarray(self):
        orig = cupy.ndarray
        with testing.AssertFunctionIsCalled('cupy.ndarray'):
            a = cupy.ndarray((2, 3), numpy.float32)
        assert cupy.ndarray is orig
        assert not isinstance(a, cupy.ndarray)

    def test_spy_ndarray(self):
        orig = cupy.ndarray
        with testing.AssertFunctionIsCalled(
                'cupy.ndarray', wraps=cupy.ndarray):
            a = cupy.ndarray((2, 3), numpy.float32)
        assert cupy.ndarray is orig
        assert isinstance(a, cupy.ndarray)

    def test_fail_not_called(self):
        orig = cupy.ndarray
        with pytest.raises(AssertionError):
            with testing.AssertFunctionIsCalled('cupy.ndarray'):
                pass
        assert cupy.ndarray is orig

    def test_fail_called_twice(self):
        orig = cupy.ndarray
        with pytest.raises(AssertionError):
            with testing.AssertFunctionIsCalled('cupy.ndarray'):
                cupy.ndarray((2, 3), numpy.float32)
                cupy.ndarray((2, 3), numpy.float32)
        assert cupy.ndarray is orig

    def test_times_called(self):
        orig = cupy.ndarray
        with testing.AssertFunctionIsCalled('cupy.ndarray', times_called=2):
            cupy.ndarray((2, 3), numpy.float32)
            cupy.ndarray((2, 3), numpy.float32)
        assert cupy.ndarray is orig

    def test_inner_error(self):
        orig = cupy.ndarray
        with pytest.raises(numpy.AxisError):
            with testing.AssertFunctionIsCalled('cupy.ndarray'):
                cupy.ndarray((2, 3), numpy.float32)
                raise numpy.AxisError('foo')
        assert cupy.ndarray is orig
