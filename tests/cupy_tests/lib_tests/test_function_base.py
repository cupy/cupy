import warnings

from cupy import testing


@testing.gpu
class TestInsert:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_almost_equal()
    def test_insert_1D_array(self, xp, dtype):
        a = testing.shaped_random((1, 5), xp, dtype=dtype)
        return xp.insert(a, 0, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_multiple(self, xp, dtype):
        a = xp.array([1, 1, 2, 2, 3, 3], dtype)
        return xp.insert(a, [2, 2], [5, 6])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_multiple_values(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        return xp.insert(a, 1, [1, 2, 3])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_multiple_indexes(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        return xp.insert(a, [1, -1, 3], 9)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_slice(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        return xp.insert(a, slice(-1, None, -1), 9)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_multiple_with_neg_index(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        return xp.insert(a, [-1, 1, 3], [7, 8, 9])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_empty(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        return xp.insert(a, [], [])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_empty_array(self, xp, dtype):
        a = xp.array([], dtype)
        return xp.insert(a, 0, [1, 2, 3])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_1D_array_boolean(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', FutureWarning)
            out = xp.insert(a, xp.array([True] * 4), 9)
            assert (w[0].category is FutureWarning)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_multi_dim_axis_none(self, xp, dtype):
        a = xp.array([[1, 1], [2, 2], [3, 3]], dtype)
        return xp.insert(a, 1, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_insert_multi_dim_with_axis(self, xp, dtype):
        a = xp.array([[1, 1], [2, 2], [3, 3]], dtype)
        return xp.insert(a, 1, 5, axis=1)
