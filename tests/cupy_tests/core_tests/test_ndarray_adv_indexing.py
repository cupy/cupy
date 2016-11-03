import unittest

import numpy as np

import cupy
from cupy import testing


@testing.gpu
class TestArrayAdvancedIndexing(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_1d_adv_indexing(self, xp, dtype):
        shp = (2, 3, 4)
        a = xp.arange(np.product(shp)).reshape(shp).astype(dtype)
        idx = (slice(None), np.array([1, 0]), slice(None))
        return a[idx]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_1d_adv_indexing_gpu_idx(self, xp, dtype):
        shp = (2, 3, 4)
        a = xp.arange(np.product(shp)).reshape(shp).astype(dtype)
        idx = (slice(None), xp.array([1, 0]), slice(None))
        return a[idx]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_2d_adv_indexing_gpu_idx(self, xp, dtype):
        shp = (2, 3, 4)
        a = xp.arange(np.product(shp)).reshape(shp).astype(dtype)
        idx = (slice(None), xp.array([[0, 1], [1, 0]]), slice(None))
        return a[idx]

    def test_not_supported_arr_and_int(self):
        shp = (2, 3, 4)
        a = cupy.arange(np.product(shp)).reshape(shp)
        idx = (slice(None), np.array([1, 0]), 1)
        with self.assertRaises(NotImplementedError):
            a[idx]

    def test_not_supported_two_arr(self):
        shp = (2, 3, 4)
        a = cupy.arange(np.product(shp)).reshape(shp)
        idx = (slice(None), np.array([1, 0]), np.array([1, 0]))
        with self.assertRaises(NotImplementedError):
            a[idx]

    def test_not_supporeted_arr_and_none(self):
        shp = (2, 3, 4)
        a = cupy.arange(np.product(shp)).reshape(shp)
        idx = (slice(None), np.array([1, 0]), None)
        with self.assertRaises(NotImplementedError):
            a[idx]
