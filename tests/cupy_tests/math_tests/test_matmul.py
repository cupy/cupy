import operator
import unittest

import numpy
import pytest

import cupy
from cupy.core import _routines_linalg as _linalg
from cupy import testing


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            # dot test
            ((3, 2), (2, 4)),
            ((3, 0), (0, 4)),
            ((0, 2), (2, 4)),
            ((3, 2), (2, 0)),
            ((2,), (2, 4)),
            ((0,), (0, 4)),
            ((3, 2), (2,)),
            ((3, 0), (0,)),
            ((2,), (2,)),
            ((0,), (0,)),
            # matmul test
            ((5, 3, 2), (5, 2, 4)),
            ((0, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2, 4)),
            ((0, 3, 2), (2, 4)),
            ((3, 2), (5, 2, 4)),
            ((3, 2), (0, 2, 4)),
            ((5, 3, 2), (1, 2, 4)),
            ((0, 3, 2), (1, 2, 4)),
            ((1, 3, 2), (5, 2, 4)),
            ((1, 3, 2), (0, 2, 4)),
            ((5, 3, 2), (2,)),
            ((5, 3, 0), (0,)),
            ((2,), (5, 2, 4)),
            ((0,), (5, 0, 4)),
            ((2, 2, 3, 2), (2, 2, 2, 4)),
            ((5, 0, 3, 2), (5, 0, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((5, 0, 3, 2), (2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((3, 2), (5, 0, 2, 4)),
            ((1, 5, 3, 2), (6, 1, 2, 4)),
            ((1, 0, 3, 2), (6, 1, 2, 4)),
            ((6, 1, 3, 2), (1, 5, 2, 4)),
            ((6, 1, 3, 2), (1, 0, 2, 4)),
            ((6, 5, 3, 2), (2,)),
            ((6, 5, 3, 0), (0,)),
            ((2,), (6, 5, 2, 4)),
            ((0,), (6, 5, 0, 4)),
            ((1, 3, 3), (10, 1, 3, 1)),
        ],
    }))
@testing.gpu
class TestMatmul(unittest.TestCase):

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((6, 5, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (6, 1, 2, 4)),
            ((6, 5, 3, 2), (1, 5, 2, 4)),
            ((6, 5, 3, 2), (1, 1, 2, 4)),
            ((6, 1, 3, 2), (6, 5, 2, 4)),
            ((1, 5, 3, 2), (6, 5, 2, 4)),
            ((1, 1, 3, 2), (6, 5, 2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((2,), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2,)),
        ],
    }))
@testing.gpu
class TestMatmulLarge(unittest.TestCase):

    # Avoid overflow
    skip_dtypes = {
        (numpy.int8, numpy.uint8),
        (numpy.int8, numpy.int16),
        (numpy.int8, numpy.float16),
        (numpy.uint8, numpy.uint8),
        (numpy.uint8, numpy.int16),
        (numpy.uint8, numpy.uint16),
        (numpy.int16, numpy.int16),
        (numpy.uint16, numpy.uint16),
    }

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        if ((dtype1, dtype2) in self.skip_dtypes or
                (dtype2, dtype1) in self.skip_dtypes):
            pytest.skip()
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        if ((dtype1, dtype2) in self.skip_dtypes or
                (dtype2, dtype1) in self.skip_dtypes):
            pytest.skip()
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_random(shape1, xp, dtype1)
        x2 = testing.shaped_random(shape2, xp, dtype2)
        return xp.matmul(x1, x2)


class TestMatmulOverflow(unittest.TestCase):

    @testing.for_int_dtypes(name='dtype', no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_overflow(self, xp, dtype):
        value = numpy.iinfo(dtype).max
        a = xp.array([value - 10]).astype(dtype)
        b = xp.array([value - 10]).astype(dtype)
        return xp.matmul(a, b)


class _TestMatmulComputeTypes(unittest.TestCase):

    def setUp(self):
        self.old_compute_type = cupy.core.get_compute_type(self.dtype)
        cupy.core.set_compute_type(self.dtype, self.compute_type)

    def tearDown(self):
        cupy.core.set_compute_type(self.dtype, self.old_compute_type)

    def make_x1_x2(self, xp, shapes, dtypes):
        x1 = testing.shaped_random(shapes[0], xp, dtypes[0])
        x2 = testing.shaped_random(shapes[1], xp, dtypes[1])
        return x1, x2


@testing.parameterize(
    *testing.product({
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair': [
            ((32, 64), (64, 96)),
            ((64, 96), (96, 32)),
            ((96, 32), (32, 64)),
        ],
    }))
@testing.gpu
class TestMatmulFp16ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float16

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, (self.dtype, self.dtype))
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, (self.dtype, self.dtype))
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
            _linalg.COMPUTE_TYPE_TF32,
        ],
        'shape_pair': [
            ((100, 200), (200, 300)),
            ((200, 300), (300, 100)),
            ((300, 100), (100, 200)),
        ],
        'dtype_pair': [
            (numpy.float16, numpy.float32),
            (numpy.float32, numpy.float32),
            (numpy.float16, numpy.complex64),
            (numpy.float32, numpy.complex64),
            (numpy.complex64, numpy.complex64),
        ],
    }))
@testing.gpu
class TestMatmulFp32ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float32

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'compute_type': [
            _linalg.COMPUTE_TYPE_DEFAULT,
            _linalg.COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair': [
            ((100, 200), (200, 300)),
            ((200, 300), (300, 100)),
            ((300, 100), (100, 200)),
        ],
        'dtype_pair': [
            (numpy.float32, numpy.float64),
            (numpy.float64, numpy.float64),
            (numpy.float32, numpy.complex128),
            (numpy.float64, numpy.complex128),
            (numpy.complex64, numpy.complex128),
            (numpy.complex128, numpy.complex128),
        ],
    }))
@testing.gpu
class TestMatmulFp64ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float64

    @testing.numpy_cupy_allclose()
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose()
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((5, 3, 1), (3, 1, 4)),
            ((3, 2, 3), (3, 2, 4)),
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
            ((3, 2), (1,)),
            ((0, 2), (3, 0)),
            ((0, 1, 1), (2, 1, 1)),
        ],
    }))
@testing.gpu
class TestMatmulInvalidShape(unittest.TestCase):

    def test_invalid_shape(self):
        for xp in (numpy, cupy):
            shape1, shape2 = self.shape_pair
            x1 = testing.shaped_arange(shape1, xp, numpy.float32)
            x2 = testing.shaped_arange(shape2, xp, numpy.float32)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)
