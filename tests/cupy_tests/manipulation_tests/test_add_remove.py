import unittest

import pytest

import cupy
from cupy import testing


class TestDelete(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_delete_with_no_axis(self, xp):
        arr = xp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        indices = xp.array([0, 2, 4, 6, 8])

        return xp.delete(arr, indices)

    @testing.numpy_cupy_array_equal()
    def test_delete_with_axis_zero(self, xp):
        arr = xp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        indices = xp.array([0, 2])

        return xp.delete(arr, indices, axis=0)

    @testing.numpy_cupy_array_equal()
    def test_delete_with_axis_one(self, xp):
        arr = xp.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indices = xp.array([0, 2, 4])

        return xp.delete(arr, indices, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_delete_with_indices_as_bool_array(self, xp):
        arr = xp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        indices = xp.array([True, False, True, False, True,
                            False, True, False, True, False])

        return xp.delete(arr, indices)

    @testing.numpy_cupy_array_equal()
    def test_delete_with_indices_as_slice(self, xp):
        arr = xp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        indices = slice(None, None, 2)
        return xp.delete(arr, indices)

    @testing.numpy_cupy_array_equal()
    def test_delete_with_indices_as_int(self, xp):
        arr = xp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        indices = 5
        if cupy.cuda.runtime.is_hip:
            pytest.xfail('HIP may have a bug')
        return xp.delete(arr, indices)


class TestAppend(unittest.TestCase):

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test(self, xp, dtype1, dtype2):
        a = testing.shaped_random((3, 4, 5), xp, dtype1)
        b = testing.shaped_random((6, 7), xp, dtype2)
        return xp.append(a, b)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_scalar_lhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype1).type(10).item()
        return xp.append(scalar, xp.arange(20, dtype=dtype2))

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_scalar_rhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype2).type(10).item()
        return xp.append(xp.arange(20, dtype=dtype1), scalar)

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_numpy_scalar_lhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype1).type(10)
        return xp.append(scalar, xp.arange(20, dtype=dtype2))

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_numpy_scalar_rhs(self, xp, dtype1, dtype2):
        scalar = xp.dtype(dtype2).type(10)
        return xp.append(xp.arange(20, dtype=dtype1), scalar)

    @testing.numpy_cupy_array_equal()
    def test_scalar_both(self, xp):
        return xp.append(10, 10)

    @testing.numpy_cupy_array_equal()
    def test_axis(self, xp):
        a = testing.shaped_random((3, 4, 5), xp, xp.float32)
        b = testing.shaped_random((3, 10, 5), xp, xp.float32)
        return xp.append(a, b, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_zerodim(self, xp):
        return xp.append(xp.array(0), xp.arange(10))

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.append(xp.array([]), xp.arange(10))


class TestResize(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test(self, xp):
        return xp.resize(xp.arange(10), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_remainder(self, xp):
        return xp.resize(xp.arange(8), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_shape_int(self, xp):
        return xp.resize(xp.arange(10), 15)

    @testing.numpy_cupy_array_equal()
    def test_scalar(self, xp):
        return xp.resize(2, (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_scalar_shape_int(self, xp):
        return xp.resize(2, 10)

    @testing.numpy_cupy_array_equal()
    def test_typed_scalar(self, xp):
        return xp.resize(xp.float32(10.0), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_zerodim(self, xp):
        return xp.resize(xp.array(0), (10, 10))

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.resize(xp.array([]), (10, 10))


class TestUnique:

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_no_axis(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, axis=1)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_index_no_axis(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_index=True)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_index(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_index=True, axis=0)[1]

    @testing.with_requires("numpy>=2.0")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_inverse_no_axis(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_inverse=True)[1]

    @testing.with_requires("numpy>=2.1")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_inverse(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_inverse=True, axis=1)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_counts_no_axis(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_counts=True)[1]

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_counts(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(a, return_counts=True, axis=0)[1]

    @testing.with_requires("numpy>=2.0")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_all_no_axis(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(
            a, return_index=True, return_inverse=True, return_counts=True)

    @testing.with_requires("numpy>=2.1")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_return_all(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique(
            a, return_index=True, return_inverse=True, return_counts=True,
            axis=1)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_empty_no_axis(self, xp, dtype):
        a = xp.empty((0,), dtype)
        return xp.unique(a)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_empty(self, xp, dtype):
        a = xp.empty((0,), dtype)
        return xp.unique(a, axis=0)

    @testing.with_requires("numpy>=2.0")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_empty_return_all_no_axis(self, xp, dtype):
        a = xp.empty((3, 0, 2), dtype)
        return xp.unique(
            a, return_index=True, return_inverse=True, return_counts=True)

    @testing.with_requires("numpy>=2.1")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_empty_return_all(self, xp, dtype):
        a = xp.empty((3, 0, 2), dtype)
        return xp.unique(
            a, return_index=True, return_inverse=True, return_counts=True,
            axis=2)

    @pytest.mark.parametrize('equal_nan', [True, False])
    @pytest.mark.parametrize('dtype', 'efdFD')
    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.23.1')
    def test_unique_equal_nan_no_axis(self, xp, dtype, equal_nan):
        if xp.dtype(dtype).kind == 'c':
            # Nan and Nan+Nan*1j are collapsed when equal_nan=True
            a = xp.array([
                complex(xp.nan, 3), 2, complex(7, xp.nan), xp.nan,
                complex(xp.nan, xp.nan), 2, xp.nan, 1
            ], dtype=dtype)
        else:
            a = xp.array([2, xp.nan, 2, xp.nan, 1], dtype=dtype)
        return xp.unique(a, equal_nan=equal_nan)

    @pytest.mark.parametrize('equal_nan', [True, False])
    @pytest.mark.parametrize('dtype', 'fdFD')
    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.23.1')
    def test_unique_equal_nan(self, xp, dtype, equal_nan):
        if xp.dtype(dtype).kind == 'c':
            # Nan and Nan+Nan*1j are collapsed when equal_nan=True
            a = xp.array([
                [complex(xp.nan, 3), 2, complex(7, xp.nan)],
                [xp.nan, complex(xp.nan, xp.nan), 2],
                [xp.nan, 1, complex(xp.nan, -1)]
            ], dtype=dtype)
        else:
            a = xp.array([
                [2, xp.nan, 2],
                [xp.nan, 1, xp.nan],
                [xp.nan, 1, xp.nan]
            ], dtype=dtype)
        return xp.unique(a, axis=0, equal_nan=equal_nan)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "attr", ["values", "indices", "inverse_indices", "counts"]
    )
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_all(self, xp, dtype, attr):
        a = testing.shaped_random((100, 100), xp, dtype)
        return getattr(xp.unique_all(a), attr)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("attr", ["values", "counts"])
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_counts(self, xp, dtype, attr):
        a = testing.shaped_random((100, 100), xp, dtype)
        return getattr(xp.unique_counts(a), attr)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("attr", ["values", "inverse_indices"])
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_inverse(self, xp, dtype, attr):
        a = testing.shaped_random((100, 100), xp, dtype)
        return getattr(xp.unique_inverse(a), attr)

    @testing.with_requires("numpy>=2.0")
    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_unique_values(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return xp.unique_values(a)


@testing.parameterize(*testing.product({
    'trim': ['fb', 'f', 'b']
}))
class TestTrim_zeros(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_non_zeros(self, xp, dtype):
        a = xp.array([-1, 2, -3, 7]).astype(dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_trimmed(self, xp, dtype):
        a = xp.array([1, 0, 2, 3, 0, 5], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_all_zeros(self, xp, dtype):
        a = xp.zeros(shape=(1000,), dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_front_zeros(self, xp, dtype):
        a = xp.array([0, 0, 4, 1, 0, 2, 3, 0, 5], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_back_zeros(self, xp, dtype):
        a = xp.array([1, 0, 2, 3, 0, 5, 0, 0, 0], dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @testing.with_requires('numpy>=2.2.0')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_zero_dim(self, xp, dtype):
        a = testing.shaped_arange((), xp, dtype)
        return xp.trim_zeros(a, trim=self.trim)

    @pytest.mark.xfail(reason='XXX: Not implemented')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trim_ndim(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype=dtype)
        return xp.trim_zeros(a, trim=self.trim)
