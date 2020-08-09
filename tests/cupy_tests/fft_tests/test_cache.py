import unittest
import pytest

import numpy as np

import cupy
from cupy import testing
from cupy.cuda import cufft
from cupy.fft import config


class TestPlanCache(unittest.TestCase):
    def setUp(self):
        config.clear_plan_cache()
        self.old_size = config.get_plan_cache_size()
        config.set_plan_cache_size(2)

    def tearDown(self):
        config.clear_plan_cache()
        config.set_plan_cache_size(self.old_size)

    def test_LRU_cache1(self):
        # test if insertion and clean-up works
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        cache.clear()
        assert cache.get_curr_size() == 0 <= cache.get_size()

    def test_LRU_cache2(self):
        # test if plan is reused
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()

    def test_LRU_cache3(self):
        # test if cache size is limited
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()
        a = testing.shaped_random((20,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        a = testing.shaped_random((30,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()

    def test_LRU_cache4(self):
        # test if fetching the plan will reorder it to the top
        cache = config.get_plan_cache()
        assert cache.get_curr_size() == 0 <= cache.get_size()

        # this creates a Plan1d
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 1 <= cache.get_size()

        # this creates a PlanNd
        a = testing.shaped_random((10, 20), cupy, cupy.float32)
        b = cupy.fft.fftn(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()

        # The first in the cache is the most recently used one;
        # using an iterator to access the linked list guarantees that
        # we don't alter the cache order
        iterator = iter(cache)
        assert isinstance(next(iterator)[1].plan, cufft.PlanNd)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)
        with pytest.raises(StopIteration):
            next(iterator)

        # this brings Plan1d to the top
        a = testing.shaped_random((10,), cupy, cupy.float32)
        b = cupy.fft.fft(a)
        assert cache.get_curr_size() == 2 <= cache.get_size()
        iterator = iter(cache)
        assert isinstance(next(iterator)[1].plan, cufft.Plan1d)
        assert isinstance(next(iterator)[1].plan, cufft.PlanNd)
        with pytest.raises(StopIteration):
            next(iterator)

        # An LRU cache guarantees that such a silly operation never
        # raises StopIteration
        iterator = iter(cache)
        for i in range(100):
            cache[next(iterator)[0]]
