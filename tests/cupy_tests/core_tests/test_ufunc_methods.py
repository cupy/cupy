import numpy

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


@testing.parameterize(*testing.product({
    'ufunc_name': ['negative', 'reciprocal']
}))
class TestUfuncAtUnary:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_at(self, xp, dtype):
        n = 50
        x = testing.shaped_random((n,), xp, dtype, seed=0) + 1
        mask = testing.shaped_random((n,), xp, bool, scale=n, seed=1)
        indices = xp.nonzero(mask)[0]
        ufunc = getattr(xp, self.ufunc_name)
        ufunc.at(x, indices)
        return x

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_at_indices(self, xp, dtype):
        n = 50
        x = testing.shaped_random((n,), xp, dtype, seed=0) + 1
        mask = testing.shaped_random((n,), xp, bool, scale=n, seed=1)
        indices = xp.nonzero(mask)[0] - n
        ufunc = getattr(xp, self.ufunc_name)
        ufunc.at(x, indices)
        return x


class TestUfuncAtBinary:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_at(self, xp, dtype):
        n = 50
        x = testing.shaped_random((n,), xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random((n,), xp=xp, dtype=bool, scale=n, seed=1)
        indices = xp.nonzero(mask)[0]
        xp.add.at(x, indices, 3)
        return x

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_at_array_values(self, xp, dtype):
        n = 50
        x = testing.shaped_random((n,), xp=xp, dtype=dtype, seed=0)
        mask = testing.shaped_random((n,), xp=xp, dtype=bool, scale=n, seed=1)
        indices = xp.nonzero(mask)[0]
        values = testing.shaped_random(
            indices.shape, xp=xp, dtype=numpy.int32, seed=2)
        xp.add.at(x, indices, values)
        return x
