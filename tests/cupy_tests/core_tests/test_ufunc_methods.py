import numpy
import pytest

import cupy
from cupy import testing


class TestUfuncOuter:

    @testing.numpy_cupy_array_equal()
    def test_add_outer(self, xp):
        x = testing.shaped_random((2, 3), xp=xp, dtype=numpy.int32, seed=0)
        y = testing.shaped_random((4, 1, 5), xp=xp, dtype=numpy.int32, seed=1)
        return xp.add.outer(x, y)

    @testing.numpy_cupy_array_equal()
    def test_add_outer_scalar(self, xp):
        return xp.add.outer(2, 3)


class TestUfuncAtAtomicOps:

    @testing.for_dtypes('iIQefd')
    @testing.numpy_cupy_array_equal()
    def test_at_add(self, xp, dtype):
        if cupy.cuda.runtime.is_hip and dtype == numpy.float16:
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (50,)
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(shape, xp=xp, dtype=bool, seed=1)
        indices = xp.nonzero(mask)[0]
        xp.add.at(x, indices, 3)
        return x

    @testing.for_dtypes('iIQfd')
    @testing.numpy_cupy_allclose()
    def test_at_min(self, xp, dtype):
        shape = (50,)
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(shape, xp=xp, dtype=bool, seed=1)
        indices = xp.nonzero(mask)[0]
        xp.minimum.at(x, indices, 3)
        return x

    @testing.for_dtypes('iIQfd')
    @testing.numpy_cupy_allclose()
    def test_at_max(self, xp, dtype):
        shape = (50,)
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(shape, xp=xp, dtype=bool, seed=1)
        indices = xp.nonzero(mask)[0]
        xp.maximum.at(x, indices, 3)
        return x

    @testing.for_dtypes('iIQefd')
    @testing.numpy_cupy_array_equal()
    def test_at_boolean_mask(self, xp, dtype):
        shape = (50,)
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(shape, xp=xp, dtype=bool, seed=1)
        xp.add.at(x, mask, 3)
        return x

    @testing.for_dtypes('iIQefd')
    @testing.numpy_cupy_array_equal()
    def test_at_array_values(self, xp, dtype):
        if cupy.cuda.runtime.is_hip and dtype == numpy.float16:
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (50,)
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(shape, xp=xp, dtype=bool, seed=1)
        indices = xp.nonzero(mask)[0]
        values = testing.shaped_random(
            indices.shape, xp=xp, dtype=numpy.int32, seed=2)
        xp.add.at(x, indices, values)
        return x

    @testing.for_dtypes('iIQefd')
    @testing.numpy_cupy_array_equal()
    def test_at_multi_dimensional(self, xp, dtype):
        if cupy.cuda.runtime.is_hip and dtype == numpy.float16:
            pytest.skip('atomicAdd does not support float16 in HIP')
        shape = (20, 30)
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(shape, xp=xp, dtype=bool, seed=1)
        indices = xp.nonzero(mask)
        xp.add.at(x, indices, 3)
        return x


class TestUfuncReduce:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-3, 'default': 1e-6})
    def test_reduce_add(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=0)
        return xp.add.reduce(x, axis=-1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-3, 'default': 1e-6})
    def test_multiply_add(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=0)
        return xp.multiply.reduce(x, axis=-1)


class TestUfuncAccumulate:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-3, 'default': 1e-6})
    def test_reduce_add(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=0)
        return xp.add.accumulate(x, axis=-1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-3, 'default': 1e-6})
    def test_multiply_add(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=0)
        return xp.multiply.accumulate(x, axis=-1)
